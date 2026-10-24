"""평가 세트 실행기 (신규) — data/eval_set.csv 로 에이전트를 측정한다

기획안: "정확도는 평가 세트로 측정해 제시" 를 수치로 만드는 도구.

평가 계층
  1) 규칙 계층 (오프라인, API 불필요)  수량·단가·수입 목적·FTA 파싱, 상업 목적 차단, 수량 한도/경계값, 외화 단가 거절
  2) HSK 추론  (live, Gemini 호출)     허용 목록 안의 정답 맞히기 + 범위 밖 품목 거절 + 프롬프트 주입 + '목록 외 코드 0건'
  3) 의도 분류 (live, Gemini 호출)     통관 / 연구실 안전 / 기타

사용 (프로젝트 루트에서):
    python src/run_eval.py --validate                 # 평가 세트 형식만 점검
    python src/run_eval.py                            # 오프라인(규칙 계층)만 실행. API 호출 없음
    python src/run_eval.py --live                     # + 의도 분류·HSK 추론 (질문당 Gemini 1회 호출)
    python src/run_eval.py --live --groups C,D --repeat 3 --sleep 2
      --groups  평가할 그룹 첫 글자 (예: A,B,C,D,E,F,G)     --repeat  같은 질문 반복 횟수(LLM 비결정성 확인)
      --sleep   호출 간 대기 초(기본 1.0)                    --limit   live 호출 대상 질문 수 제한

결과: output/eval/eval_report_<시각>.md (요약·실패 사례) 와 eval_results_<시각>.csv (건별 결과)
      - 정확도에는 Wilson 95% 신뢰구간을 함께 표시한다 (표본이 작아서 점추정만으로는 오해하기 쉽다)
      - API 오류(429·서버 혼잡 등)로 답을 못 받은 건은 정확도에서 제외하고 따로 센다
      - 429(사용량 초과)가 나오면 즉시 중단하고 지금까지의 결과로 보고서를 만든다
"""
import argparse
import contextlib
import csv
import datetime
import io
import math
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import report_logic as rl  # noqa: E402

EVAL_PATH = config.DATA_DIR / "eval_set.csv"
EVAL_OUT_DIR = config.OUTPUT_DIR / "eval"

COLUMNS = ["id", "group", "query", "expected_route", "expected_hsk", "hint_hsk", "expected_status",
           "expected_stage", "expected_purpose", "expected_qty", "expected_price", "expected_fta",
           "difficulty", "notes"]
ROUTES = ("customs", "lab_safety", "other")
NONE = "NONE"  # 품목표 밖(거절되어야 함)


# ── 로드·검증 ───────────────────────────────────────────────────────────────
def load_eval_set(path=None) -> list:
    with open(path or EVAL_PATH, encoding="utf-8-sig", newline="") as handle:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(handle)]


def validate_eval_set(rows: list, catalog: dict) -> list:
    """형식 오류 메시지 목록 (비어 있으면 정상)."""
    errors = []
    if not rows:
        return ["평가 세트가 비어 있습니다."]
    missing = [c for c in COLUMNS if c not in rows[0]]
    if missing:
        return [f"필수 열이 없습니다: {', '.join(missing)}"]

    seen = set()
    for row in rows:
        rid = row["id"]
        where = f"[{rid or '?'}]"
        if not rid or rid in seen:
            errors.append(f"{where} id 가 비었거나 중복입니다.")
        seen.add(rid)
        if not row["group"] or not row["query"]:
            errors.append(f"{where} group/query 가 비었습니다.")
        if row["expected_route"] not in ("",) + ROUTES:
            errors.append(f"{where} expected_route 값 오류: {row['expected_route']}")
        if row["expected_hsk"] not in ("", NONE) and row["expected_hsk"] not in catalog:
            errors.append(f"{where} expected_hsk 가 품목표에 없습니다: {row['expected_hsk']}")
        if row["hint_hsk"] and row["hint_hsk"] not in catalog:
            errors.append(f"{where} hint_hsk 가 품목표에 없습니다: {row['hint_hsk']}")
        if row["expected_status"] not in ("", "success", "error"):
            errors.append(f"{where} expected_status 값 오류: {row['expected_status']}")
        if row["expected_stage"] not in ("", "report", "done"):
            errors.append(f"{where} expected_stage 값 오류: {row['expected_stage']}")
        if bool(row["expected_status"]) != bool(row["expected_stage"]):
            errors.append(f"{where} expected_status 와 expected_stage 는 함께 지정해야 합니다.")
        if row["expected_status"] and not (row["hint_hsk"] or row["expected_hsk"] in catalog):
            errors.append(f"{where} 규칙 게이트 평가에는 hint_hsk(품목 코드)가 필요합니다.")
        if row["expected_purpose"] not in ("", "research", "commercial", "unknown"):
            errors.append(f"{where} expected_purpose 값 오류: {row['expected_purpose']}")
        if row["expected_qty"] not in ("", NONE) and not row["expected_qty"].isdigit():
            errors.append(f"{where} expected_qty 값 오류: {row['expected_qty']}")
        if row["expected_price"] not in ("", "missing", "foreign") and not row["expected_price"].isdigit():
            errors.append(f"{where} expected_price 값 오류: {row['expected_price']}")
        if row["expected_fta"] not in ("", "yes", "no"):
            errors.append(f"{where} expected_fta 값 오류: {row['expected_fta']}")
        if row["difficulty"] not in ("", "easy", "medium", "hard"):
            errors.append(f"{where} difficulty 값 오류: {row['difficulty']}")
    return errors


# ── 1) 규칙 계층 (오프라인) ───────────────────────────────────────────────────
def _check(row, name, expected, actual) -> dict:
    return {"id": row["id"], "group": row["group"], "query": row["query"], "check": name,
            "expected": str(expected), "actual": str(actual), "ok": str(expected) == str(actual)}


def run_gate(query: str, hint_hsk: str) -> tuple:
    """보고서 단계의 규칙 게이트만 실행한다 (LLM·벡터 DB는 가짜로 대체, 파일은 임시 폴더에 생성)."""
    import main_pipeline as mp

    with tempfile.TemporaryDirectory() as tmp, \
            mock.patch.object(mp, "OUTPUT_DIR", Path(tmp)), \
            mock.patch.object(mp, "retrieve_context", return_value=("", [])), \
            mock.patch.object(mp.config, "generate_text", side_effect=RuntimeError("eval stub")), \
            contextlib.redirect_stdout(io.StringIO()):
        report = mp.generate_fast_track_report(query, hint_hsk)
    status = report.get("status")
    return status, ("done" if status == "success" else "report")


def evaluate_rules(rows: list, catalog: dict) -> list:
    checks = []
    for row in rows:
        query = row["query"]
        quantities = rl.find_quantities(query)
        quantity = quantities[0] if quantities else None

        if row["expected_purpose"]:
            checks.append(_check(row, "수입 목적", row["expected_purpose"], rl.detect_purpose(query)))
        if row["expected_qty"]:
            expected = None if row["expected_qty"] == NONE else int(row["expected_qty"])
            checks.append(_check(row, "수량 추출", expected, quantity))
        if row["expected_price"]:
            price = rl.extract_unit_price(query, quantity)
            actual = price["unit_krw"] if price["status"] == "found" else price["status"]
            checks.append(_check(row, "단가 추출", row["expected_price"], actual))
        if row["expected_fta"]:
            checks.append(_check(row, "원산지증명서(FTA)", row["expected_fta"], "yes" if rl.mentions_origin_cert(query) else "no"))
        if row["expected_status"]:
            hint = row["hint_hsk"] or row["expected_hsk"]
            status, stage = run_gate(query, hint)
            checks.append(_check(row, "게이트 결과", f"{row['expected_status']}/{row['expected_stage']}", f"{status}/{stage}"))
    return checks


# ── 2·3) live: HSK 추론, 의도 분류 ────────────────────────────────────────────
def classify_hsk(result: dict, expected: str, catalog: dict) -> dict:
    """infer_hsk_code 결과를 평가용 outcome 으로 바꾼다.

    in-scope(정답 코드가 있음): correct | wrong(다른 품목) | missed(품목 특정 못 함) | abstain(신뢰도 부족 보류) | blocked(목록 외 코드를 차단)
    out-of-scope(NONE)        : correct_reject(거절 성공) | false_accept(엉뚱한 품목으로 통과)
    공통                      : api_error | parse_error
    """
    out = {"predicted": "", "confidence": "", "note": "", "blocked_event": False, "violation": False}
    if result.get("status") == "success":
        data = result["data"]
        out.update(predicted=data["hsk_code"], confidence=f"{data['confidence']:.2f}")
        out["violation"] = data["hsk_code"] not in catalog
        out["outcome"] = "false_accept" if expected == NONE else ("correct" if data["hsk_code"] == expected else "wrong")
        return out

    message = result.get("message", "")
    out["note"] = message[:120]
    if result.get("error_kind"):
        out["outcome"], out["note"] = "api_error", f"{result['error_kind']}: {message[:100]}"
    elif "candidate" in result:
        cand = result["candidate"]
        out.update(predicted=cand["hsk_code"], confidence=f"{cand['confidence']:.2f}")
        out["outcome"] = "correct_reject" if expected == NONE else "abstain"
    elif message.startswith("품목표에 없는 HSK 코드"):
        out["blocked_event"] = True
        out["outcome"] = "correct_reject" if expected == NONE else "blocked"
    elif message.startswith("지원 대상 품목을 특정하지 못했습니다"):
        out["outcome"] = "correct_reject" if expected == NONE else "missed"
    elif "해석하지 못했습니다" in message:
        out["outcome"] = "parse_error"
    else:
        out["outcome"] = "api_error"
    return out


def classify_route(route, expected: str) -> dict:
    from intent_router import CUSTOMS, GENERAL, LAB_SAFETY, UNKNOWN

    accepted = {"customs": (CUSTOMS,), "lab_safety": (LAB_SAFETY,), "other": (GENERAL, UNKNOWN)}[expected]
    out = {"predicted": route.category, "confidence": f"{route.confidence:.2f}", "note": route.reason[:120],
           "blocked_event": False, "violation": False}
    if route.fallback_used:
        out["outcome"] = "api_error"  # LLM 호출이 실패해 키워드 규칙으로 대체된 건은 LLM 정확도에서 제외
    else:
        out["outcome"] = "correct" if route.category in accepted else "wrong"
    return out


def run_live(rows: list, catalog: dict, sleep_s: float, repeat: int, limit: int, log=print) -> tuple:
    """(결과 목록, 중단 사유 또는 None). 429 가 나오면 즉시 중단한다."""
    from hsk_inferencer import infer_hsk_code
    from intent_router import IntentRouter

    router = IntentRouter()  # GEMINI_API_KEY 가 없으면 RuntimeError
    targets = [r for r in rows if r["expected_route"] or r["expected_hsk"]]
    if limit:
        targets = targets[:limit]

    results, stopped = [], None
    for index, row in enumerate(targets, 1):
        for trial in range(1, repeat + 1):
            if row["expected_route"]:
                route = router.route(row["query"])
                rec = classify_route(route, row["expected_route"])
                rec.update(kind="route", expected=row["expected_route"])
                results.append({**_base(row, trial), **rec})
                time.sleep(sleep_s)
                if rec["outcome"] == "api_error" and "사용량 한도" in rec["note"]:
                    stopped = "429(사용량 한도 초과)로 의도 분류 중단"
            if not stopped and row["expected_hsk"]:
                rec = classify_hsk(infer_hsk_code(row["query"]), row["expected_hsk"], catalog)
                rec.update(kind="hsk", expected=row["expected_hsk"])
                results.append({**_base(row, trial), **rec})
                time.sleep(sleep_s)
                if rec["outcome"] == "api_error" and rec["note"].startswith("quota"):
                    stopped = "429(사용량 한도 초과)로 HSK 추론 중단"
            if stopped:
                break
        log(f"  [{index}/{len(targets)}] {row['id']} 완료")
        if stopped:
            log(f"  ⚠ {stopped} — 지금까지의 결과로 보고서를 만듭니다.")
            break
    return results, stopped


def _base(row, trial):
    return {"id": row["id"], "group": row["group"], "query": row["query"], "trial": trial, "difficulty": row["difficulty"]}


# ── 집계 ────────────────────────────────────────────────────────────────────
def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """이항 비율의 Wilson 95% 신뢰구간."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _pct(k: int, n: int) -> str:
    if n == 0:
        return "-"
    low, high = wilson(k, n)
    return f"{k / n * 100:.1f}% ({k}/{n}, 95% CI {low * 100:.0f}~{high * 100:.0f}%)"


def summarize_rules(checks: list) -> dict:
    by_check = defaultdict(lambda: [0, 0])
    for c in checks:
        by_check[c["check"]][1] += 1
        by_check[c["check"]][0] += c["ok"]
    return {"total": len(checks), "passed": sum(c["ok"] for c in checks), "by_check": dict(by_check),
            "failed": [c for c in checks if not c["ok"]], "all": checks}


def summarize_live(results: list) -> dict:
    hsk = [r for r in results if r["kind"] == "hsk"]
    route = [r for r in results if r["kind"] == "route"]
    valid_hsk = [r for r in hsk if r["outcome"] not in ("api_error", "parse_error")]
    in_scope = [r for r in valid_hsk if r["expected"] != NONE]
    oos = [r for r in valid_hsk if r["expected"] == NONE]
    valid_route = [r for r in route if r["outcome"] != "api_error"]

    groups = defaultdict(lambda: [0, 0])
    for r in in_scope:
        groups[r["group"]][1] += 1
        groups[r["group"]][0] += r["outcome"] == "correct"
    routes = defaultdict(lambda: [0, 0])
    for r in valid_route:
        routes[r["expected"]][1] += 1
        routes[r["expected"]][0] += r["outcome"] == "correct"

    return {
        "hsk_trials": len(hsk), "hsk_excluded": len(hsk) - len(valid_hsk),
        "in_scope_n": len(in_scope), "in_scope_correct": sum(r["outcome"] == "correct" for r in in_scope),
        "in_scope_wrong": sum(r["outcome"] == "wrong" for r in in_scope),
        "in_scope_abstain": sum(r["outcome"] in ("abstain", "missed") for r in in_scope),
        "in_scope_blocked": sum(r["outcome"] == "blocked" for r in in_scope),
        "oos_n": len(oos), "oos_rejected": sum(r["outcome"] == "correct_reject" for r in oos),
        "blocked_events": sum(1 for r in valid_hsk if r["blocked_event"]),
        "violations": sum(1 for r in hsk if r["violation"]),
        "groups": dict(groups),
        "route_trials": len(route), "route_excluded": len(route) - len(valid_route),
        "route_n": len(valid_route), "route_correct": sum(r["outcome"] == "correct" for r in valid_route),
        "routes": dict(routes),
        "failures": [r for r in results if r["outcome"] in ("wrong", "missed", "abstain", "blocked", "false_accept")
                     or (r["kind"] == "route" and r["outcome"] == "wrong")],
        "api_errors": [r for r in results if r["outcome"] in ("api_error", "parse_error")],
    }


# ── 보고서 ──────────────────────────────────────────────────────────────────
def build_report(rows, rules, live, stopped, args) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# 평가 보고서 ({now})", "",
             f"- 평가 세트: {len(rows)}문항 (그룹: {', '.join(f'{g} {n}' for g, n in sorted(Counter(r['group'] for r in rows).items()))})",
             f"- 모드: {'오프라인 + live(Gemini)' if live else '오프라인(규칙 계층만, API 호출 없음)'}"]
    if live:
        lines.append(f"- 모델: {config.MODEL_NAME} / 반복 {args.repeat}회 / 호출 간 대기 {args.sleep}초")
    if stopped:
        lines.append(f"- ⚠ **중단됨**: {stopped} (부분 결과)")
    lines += ["", "## 1. 규칙 계층 (결정적 로직)", "",
              f"- 전체 통과: {_pct(rules['passed'], rules['total'])}", "",
              "| 검사 | 통과 |", "|:---|:---|"]
    for name, (ok, total) in sorted(rules["by_check"].items()):
        lines.append(f"| {name} | {_pct(ok, total)} |")

    if live:
        s = summarize_live(live)
        lines += ["", "## 2. HSK 추론 (허용 목록 기반)", "",
                  "| 지표 | 값 |", "|:---|:---|",
                  f"| 품목 정확도 (목록 안 정답 맞힘) | {_pct(s['in_scope_correct'], s['in_scope_n'])} |",
                  f"| 오답 (다른 품목으로 확정) | {s['in_scope_wrong']}건 |",
                  f"| 보류·미특정 (신뢰도 부족/품목 못 찾음) | {s['in_scope_abstain']}건 |",
                  f"| 목록 외 코드 생성 후 차단됨 | {s['in_scope_blocked']}건 (안전한 오답) |",
                  f"| 범위 밖 품목 거절률 | {_pct(s['oos_rejected'], s['oos_n'])} |",
                  f"| **목록 외 코드가 결과로 나온 건수** | **{s['violations']}건** (기획안 기준: 0건) |",
                  f"| 환각 시도 차단 이벤트 | {s['blocked_events']}건 |",
                  f"| API 오류로 제외 | {s['hsk_excluded']}건 / 전체 {s['hsk_trials']}회 |", "",
                  "그룹별 품목 정확도", "", "| 그룹 | 정확도 |", "|:---|:---|"]
        for group, (ok, total) in sorted(s["groups"].items()):
            lines.append(f"| {group} | {_pct(ok, total)} |")
        lines += ["", "## 3. 의도 분류", "", f"- 전체: {_pct(s['route_correct'], s['route_n'])}  (LLM 실패로 규칙 대체된 {s['route_excluded']}건 제외)", "",
                  "| 정답 클래스 | 정확도 |", "|:---|:---|"]
        for name, (ok, total) in sorted(s["routes"].items()):
            lines.append(f"| {name} | {_pct(ok, total)} |")

    lines += ["", "## 4. 실패 사례", ""]
    failed_rows = [(c["id"], c["check"], c["query"], c["expected"], c["actual"], "") for c in rules["failed"]]
    if live:
        failed_rows += [(r["id"], f"{r['kind']}/{r['outcome']}", r["query"], r["expected"], r["predicted"] or "-", r["note"])
                        for r in summarize_live(live)["failures"]]
    if failed_rows:
        lines += ["| id | 구분 | 질문 | 기대 | 실제 | 비고 |", "|:---|:---|:---|:---|:---|:---|"]
        for rid, kind, query, exp, act, note in failed_rows:
            lines.append(f"| {rid} | {kind} | {query[:60]} | {exp} | {act} | {note[:60]} |")
    else:
        lines.append("실패 사례 없음.")

    lines += ["", "## 5. 해석 시 주의", "",
              "- 정답 라벨은 작성자의 판단입니다. 난이도 `hard` 중 일부(C05·C06 충전기 등)는 분류 근거가 갈릴 수 있으니 실패 사례를 보고 라벨을 재검토하세요.",
              "- LLM 응답은 매번 달라질 수 있습니다. `--repeat 3` 이상으로 안정성을 확인하세요.",
              "- 표본이 작아 신뢰구간이 넓습니다. 점추정값 하나만 인용하지 말고 구간을 함께 제시하세요.",
              "- 품목표(현재 5개 품목)에 의존합니다. 품목이나 한도를 바꾸면 평가 세트 라벨도 함께 갱신해야 합니다."]
    return "\n".join(lines) + "\n"


def write_outputs(report: str, rules: dict, live: list, out_dir: Path) -> tuple:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path, csv_path = out_dir / f"eval_report_{stamp}.md", out_dir / f"eval_results_{stamp}.csv"
    md_path.write_text(report, encoding="utf-8")
    fields = ["kind", "id", "group", "difficulty", "trial", "query", "expected", "predicted", "confidence", "outcome", "note"]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{"kind": "rule/" + c["check"], "id": c["id"], "group": c["group"], "query": c["query"],
                           "expected": c["expected"], "predicted": c["actual"], "outcome": "ok" if c["ok"] else "fail"}
                          for c in rules["all"]])
        writer.writerows(live)
    return md_path, csv_path


# ── CLI ─────────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="평가 세트 실행기")
    parser.add_argument("--validate", action="store_true", help="평가 세트 형식만 점검")
    parser.add_argument("--live", action="store_true", help="Gemini 를 호출해 의도 분류·HSK 추론도 평가")
    parser.add_argument("--groups", default="", help="평가할 그룹 첫 글자(쉼표 구분). 예: A,B,C")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    from tariff_catalog import CatalogError, get_catalog

    try:
        catalog = get_catalog()
        rows = load_eval_set()
    except (CatalogError, OSError) as exc:
        print(f"[오류] {exc}")
        return 2

    errors = validate_eval_set(rows, catalog)
    if errors:
        print("[평가 세트 오류]")
        for message in errors:
            print(" -", message)
        return 2
    print(f"평가 세트 {len(rows)}문항 확인 (품목표 {len(catalog)}개 품목)")
    if args.validate:
        return 0

    if args.groups:
        wanted = tuple(g.strip().upper() for g in args.groups.split(",") if g.strip())
        rows = [r for r in rows if r["group"].upper().startswith(wanted)]
        print(f"그룹 필터 {wanted}: {len(rows)}문항")

    print("1. 규칙 계층 평가 (오프라인)...")
    checks = evaluate_rules(rows, catalog)
    rules = summarize_rules(checks)
    print(f"   통과 {rules['passed']}/{rules['total']}")

    live, stopped = [], None
    if args.live:
        print("2. live 평가 (Gemini 호출)...")
        try:
            live, stopped = run_live(rows, catalog, args.sleep, max(1, args.repeat), args.limit)
        except RuntimeError as exc:  # 키 없음
            print(f"[오류] {exc}")
            return 2

    report = build_report(rows, rules, live, stopped, args)
    md_path, csv_path = write_outputs(report, rules, live, EVAL_OUT_DIR)
    print(f"\n보고서: {md_path}\n건별 결과: {csv_path}")
    violations = summarize_live(live)["violations"] if live else 0
    return 0 if (rules["passed"] == rules["total"] and violations == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
