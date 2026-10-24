"""평가 이어하기·합산 도구 (run_eval_resume.py) — run_eval.py 는 수정하지 않고 그 함수들을 가져다 쓴다

왜 필요한가
  무료 한도(텍스트 모델 하루 20회·분당 5회)로는 live 평가 61회를 하루에 끝낼 수 없다.
  이 도구는
    1) 이미 끝난 문항은 건너뛰고            (이어하기)
    2) 오늘 쓸 호출 수(--budget)만큼만 실행하고 (한도 보호)
    3) 모든 날의 결과를 합쳐 보고서 하나를 만든다 (합산)

사용 (프로젝트 루트에서)
    python src/run_eval_resume.py --status                 # 진행 현황 (API 호출 0회)
    python src/run_eval_resume.py --merge-only             # 누적 결과로 보고서만 생성 (API 호출 0회)
    python src/run_eval_resume.py --run                    # 남은 문항을 이어서 실행 (기본 18회, 15초 간격)
    python src/run_eval_resume.py --run --groups C,D --budget 19
        --groups  이번에 실행할 그룹과 순서(기본 D,G,C,A,B,E: 기획서 핵심 지표부터)
        --budget  이번 실행의 최대 호출 수(기본 18 = 일일 한도 20 - 여유 2, 0이면 제한 없음)
        --sleep   호출 간 대기 초(기본 15 = 분당 5회 한도 대응)
        --repeat  같은 질문 반복 횟수(기본 1)

동작 규칙
  * 결과는 호출마다 output/eval/eval_results_<시각>_resume.csv 에 바로 저장한다 (중간에 끊겨도 보존).
  * output/eval/eval_results_*.csv 를 전부 읽어 합친다. run_eval.py 가 만든 예전 결과도 포함된다.
  * 같은 (종류, id, 시도) 결과가 여럿이면 최신 '유효' 결과를 쓴다. API 오류 결과는 유효 결과를 덮어쓰지 않고, 다시 실행 대상이 된다.
  * 질문이나 정답 라벨이 지금의 eval_set.csv 와 다른 예전 결과, 다른 모델로 얻은 결과는 합산에서 제외하고 알려 준다.
  * 429(사용량 한도) 또는 API 키 오류가 나오면 즉시 멈춘다. 서버 혼잡 등 다른 API 오류가 3번 연속 나와도 멈춘다.
  * 합산 보고서: output/eval/eval_report_merged_<시각>.md, 건별 합산: eval_merged_<시각>.csv
    (이 두 파일은 다음 합산 때 다시 읽히지 않는다)
"""
import argparse
import csv
import datetime
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_eval  # noqa: E402

DEFAULT_ORDER = ("D", "G", "C", "A", "B", "E")
DEFAULT_BUDGET = 18
DEFAULT_SLEEP = 15.0
RESULT_GLOB = "eval_results_*.csv"
MAX_CONSECUTIVE_ERRORS = 3
FIELDS = ["kind", "id", "group", "difficulty", "trial", "query", "expected", "predicted", "confidence",
          "outcome", "note", "blocked_event", "violation", "model", "timestamp"]
_TRUE = {"1", "true", "yes", "y"}


# ── 대상·순서 ───────────────────────────────────────────────────────────────
def group_key(row: dict) -> str:
    return row["group"][:1].upper()


def resolve_order(groups_arg: str, rows: list) -> list:
    """--groups 가 있으면 그 순서대로, 없으면 DEFAULT_ORDER 뒤에 나머지 그룹을 붙인다."""
    if groups_arg:
        return [g.strip().upper() for g in groups_arg.split(",") if g.strip()]
    present = sorted({group_key(r) for r in rows})
    return list(DEFAULT_ORDER) + [g for g in present if g not in DEFAULT_ORDER]


def build_targets(rows: list, order: list, repeat: int) -> list:
    """[(kind, row, trial)] — 문항 순서대로 의도 분류(route) 다음에 HSK 추론(hsk). 각 항목이 API 1회."""
    targets = []
    for group in order:
        for row in (r for r in rows if group_key(r) == group):
            for trial in range(1, repeat + 1):
                if row["expected_route"]:
                    targets.append(("route", row, trial))
                if row["expected_hsk"]:
                    targets.append(("hsk", row, trial))
    return targets


def _key(kind, row_id, trial):
    return (kind, row_id, trial)


# ── 결과 읽기·합치기 ─────────────────────────────────────────────────────────
def _as_bool(value) -> bool:
    return str(value).strip().lower() in _TRUE


def _is_valid(result: dict) -> bool:
    return result["outcome"] != "api_error"


def row_to_result(raw: dict, catalog: dict):
    """CSV 한 줄을 summarize_live 가 쓰는 결과 dict 로 되돌린다. run_eval.py 가 만든 예전 CSV(blocked_event·violation 열 없음)도 복원한다."""
    kind = raw.get("kind") or ""
    if kind not in ("route", "hsk"):
        return None  # rule/... 줄은 매번 오프라인으로 다시 계산하므로 읽지 않는다
    outcome = raw.get("outcome") or ""
    predicted = raw.get("predicted") or ""
    note = raw.get("note") or ""

    if (raw.get("violation") or "") != "":
        violation = _as_bool(raw["violation"])
    else:  # 예전 CSV: 확정된 코드가 품목표에 없으면 위반
        violation = (kind == "hsk" and outcome in ("correct", "wrong", "false_accept")
                     and bool(predicted) and predicted not in catalog)
    if (raw.get("blocked_event") or "") != "":
        blocked = _as_bool(raw["blocked_event"])
    else:  # 예전 CSV: 차단 안내 문구로 복원
        blocked = kind == "hsk" and note.startswith("품목표에 없는 HSK 코드")

    try:
        trial = int(raw.get("trial") or 1)
    except ValueError:
        trial = 1
    return {
        "kind": kind, "id": raw.get("id") or "", "group": raw.get("group") or "",
        "difficulty": raw.get("difficulty") or "", "trial": trial, "query": raw.get("query") or "",
        "expected": raw.get("expected") or "", "predicted": predicted, "confidence": raw.get("confidence") or "",
        "outcome": outcome, "note": note, "blocked_event": blocked, "violation": violation,
        "model": raw.get("model") or "", "timestamp": raw.get("timestamp") or "",
    }


def load_results(out_dir: Path, rows: list, catalog: dict, model: str) -> tuple:
    """(결과 목록, 집계 Counter, 읽은 파일 수). 오래된 파일부터 읽어 나중 것이 이기되, API 오류는 유효 결과를 덮어쓰지 않는다."""
    by_id = {r["id"]: r for r in rows}
    best, stats = {}, Counter()
    files = sorted(Path(out_dir).glob(RESULT_GLOB)) if Path(out_dir).exists() else []
    for path in files:
        with open(path, encoding="utf-8-sig", newline="") as handle:
            for raw in csv.DictReader(handle):
                result = row_to_result(raw, catalog)
                if result is None:
                    continue
                row = by_id.get(result["id"])
                if row is None:
                    stats["stale"] += 1
                    continue
                label = row["expected_route"] if result["kind"] == "route" else row["expected_hsk"]
                if result["query"] != row["query"] or result["expected"] != label:
                    stats["stale"] += 1  # 질문이나 정답 라벨이 바뀐 뒤에는 예전 결과를 쓰지 않는다
                    continue
                if result["model"] and result["model"] != model:
                    stats["other_model"] += 1
                    continue
                key = _key(result["kind"], result["id"], result["trial"])
                previous = best.get(key)
                if previous is None or _is_valid(result) or not _is_valid(previous):
                    best[key] = result
    for result in best.values():
        if _is_valid(result) and not result["model"]:
            stats["unknown_model"] += 1
    return list(best.values()), stats, len(files)


def progress(targets: list, results: list) -> tuple:
    """({그룹: [완료, 전체]}, 미완료 대상 목록)"""
    done = {_key(r["kind"], r["id"], r["trial"]) for r in results if _is_valid(r)}
    per_group, pending = defaultdict(lambda: [0, 0]), []
    for kind, row, trial in targets:
        slot = per_group[group_key(row)]
        slot[1] += 1
        if _key(kind, row["id"], trial) in done:
            slot[0] += 1
        else:
            pending.append((kind, row, trial))
    return dict(per_group), pending


# ── 실행 ────────────────────────────────────────────────────────────────────
def _unique_path(path: Path) -> Path:
    candidate, n = path, 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{n}{path.suffix}")
        n += 1
    return candidate


class ProgressFile:
    """호출마다 한 줄씩 바로 저장해서 중간에 끊겨도 결과가 남는다. 첫 결과가 생길 때 파일을 만든다."""

    def __init__(self, out_dir: Path):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = _unique_path(Path(out_dir) / f"eval_results_{stamp}_resume.csv")
        self._handle = None
        self._writer = None

    def write(self, result: dict) -> None:
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = open(self.path, "w", encoding="utf-8-sig", newline="")
            self._writer = csv.DictWriter(self._handle, fieldnames=FIELDS, extrasaction="ignore")
            self._writer.writeheader()
        row = dict(result)
        row["blocked_event"] = int(bool(row.get("blocked_event")))
        row["violation"] = int(bool(row.get("violation")))
        self._writer.writerow(row)
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def _get_clients(need_router: bool = True) -> tuple:
    """(IntentRouter 또는 None, infer_hsk_code). 키가 없으면 RuntimeError."""
    from hsk_inferencer import infer_hsk_code

    router = None
    if need_router:
        from intent_router import IntentRouter

        router = IntentRouter()
    return router, infer_hsk_code


def _is_quota_note(note: str) -> bool:
    return "사용량 한도" in note or note.startswith("quota")


def run_pending(pending: list, catalog: dict, out_dir: Path, budget: int, sleep_s: float, model: str, log=print) -> tuple:
    """(실행한 호출 수, 중단 사유 또는 None). 호출마다 결과를 바로 저장한다."""
    todo = pending if budget <= 0 else pending[:budget]
    if not todo:
        return 0, None
    router, infer = _get_clients(need_router=any(kind == "route" for kind, _, _ in todo))
    progress_file = ProgressFile(out_dir)
    stopped, consecutive, done = None, 0, 0
    try:
        for index, (kind, row, trial) in enumerate(todo, 1):
            if kind == "route":
                rec = run_eval.classify_route(router.route(row["query"]), row["expected_route"])
                rec.update(kind="route", expected=row["expected_route"])
            else:
                rec = run_eval.classify_hsk(infer(row["query"]), row["expected_hsk"], catalog)
                rec.update(kind="hsk", expected=row["expected_hsk"])
            result = {**run_eval._base(row, trial), **rec, "model": model,
                      "timestamp": datetime.datetime.now().isoformat(timespec="seconds")}
            progress_file.write(result)
            done += 1
            log(f"  [{index}/{len(todo)}] {row['id']} {kind} → {rec['outcome']}")

            if rec["outcome"] == "api_error":
                note = rec["note"]
                if _is_quota_note(note):
                    stopped = "429(사용량 한도 초과)로 중단"
                    break
                if note.startswith("config"):
                    stopped = "API 키 설정 오류로 중단"
                    break
                consecutive += 1
                if consecutive >= MAX_CONSECUTIVE_ERRORS:
                    stopped = f"API 오류가 {MAX_CONSECUTIVE_ERRORS}번 연속 발생해 중단"
                    break
            else:
                consecutive = 0
            if index < len(todo):
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        stopped = "사용자 중단(Ctrl+C)"
    finally:
        progress_file.close()
    if done:
        log(f"  → 결과 저장: {progress_file.path.name}")
    return done, stopped


# ── 합산 보고서 ──────────────────────────────────────────────────────────────
def progress_block(per_group: dict, pending: list, stats: Counter, n_files: int, model: str) -> str:
    total = sum(t for _, t in per_group.values())
    done = sum(d for d, _ in per_group.values())
    lines = ["", "## 0. 누적 진행 현황", "",
             f"- 병합한 결과 파일: {n_files}개 (output/eval/eval_results_*.csv)",
             f"- 모델: {model}" + (f" (모델 기록이 없는 이전 결과 {stats['unknown_model']}건은 같은 모델로 간주)" if stats["unknown_model"] else ""),
             f"- 완료 호출: {done} / {total}" + (f" ({done / total * 100:.0f}%)" if total else ""), "",
             "| 그룹 | 완료/전체 |", "|:---|:---|"]
    for group, (d, t) in sorted(per_group.items()):
        lines.append(f"| {group} | {d}/{t} |")
    if pending:
        ids = ", ".join(f"{row['id']}({kind})" for kind, row, _ in pending[:30])
        lines += ["", f"- 아직 안 한 항목 {len(pending)}건: {ids}" + (" …" if len(pending) > 30 else ""),
                  "- ⚠ 미완료 상태의 수치는 일부 문항만 반영한 값입니다. 인용할 때 문항 수(n)를 함께 밝히세요."]
    else:
        lines += ["", "- 모든 대상 문항이 완료되었습니다."]
    excluded = []
    if stats["stale"]:
        excluded.append(f"질문·정답 라벨이 바뀐 예전 결과 {stats['stale']}건")
    if stats["other_model"]:
        excluded.append(f"다른 모델로 얻은 결과 {stats['other_model']}건")
    if excluded:
        lines.append("- 합산에서 제외: " + ", ".join(excluded))
    return "\n".join(lines) + "\n"


def write_merged(report: str, results: list, out_dir: Path) -> tuple:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = _unique_path(out_dir / f"eval_report_merged_{stamp}.md")
    csv_path = _unique_path(out_dir / f"eval_merged_{stamp}.csv")  # eval_results_*.csv 가 아니라서 다음 합산에 다시 읽히지 않는다
    md_path.write_text(report, encoding="utf-8")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for result in sorted(results, key=lambda r: (r["id"], r["kind"], r["trial"])):
            row = dict(result)
            row["blocked_event"] = int(bool(row["blocked_event"]))
            row["violation"] = int(bool(row["violation"]))
            writer.writerow(row)
    return md_path, csv_path


# ── CLI ─────────────────────────────────────────────────────────────────────
def _load_catalog() -> dict:
    from tariff_catalog import get_catalog

    return get_catalog()


def _load_rows() -> list:
    return run_eval.load_eval_set()


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="평가 이어하기·합산 도구")
    parser.add_argument("--status", action="store_true", help="진행 현황만 출력 (API 호출 0회)")
    parser.add_argument("--merge-only", action="store_true", help="누적 결과로 보고서만 생성 (API 호출 0회)")
    parser.add_argument("--run", action="store_true", help="남은 문항을 이어서 실행 (Gemini 호출)")
    parser.add_argument("--groups", default="", help="이번에 실행할 그룹과 순서. 예: C,D")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET, help="이번 실행의 최대 호출 수 (0=제한 없음)")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP, help="호출 간 대기 초")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args(argv)
    args.repeat = max(1, args.repeat)

    try:
        catalog = _load_catalog()
        rows = _load_rows()
    except Exception as exc:  # noqa: BLE001 - 품목표(CatalogError)·평가 세트 파일 오류를 메시지로 알린다
        print(f"[오류] {exc}")
        return 2
    errors = run_eval.validate_eval_set(rows, catalog)
    if errors:
        print("[평가 세트 오류]")
        for message in errors:
            print(" -", message)
        return 2

    out_dir = Path(run_eval.EVAL_OUT_DIR)
    model = run_eval.config.MODEL_NAME
    full_order = resolve_order("", rows)
    run_order = resolve_order(args.groups, rows)
    all_targets = build_targets(rows, full_order, args.repeat)

    results, stats, n_files = load_results(out_dir, rows, catalog, model)
    per_group, pending_all = progress(all_targets, results)
    pending_run = [t for t in pending_all if group_key(t[1]) in run_order]
    pending_run.sort(key=lambda t: run_order.index(group_key(t[1])))  # 지정한 그룹 순서대로

    print(f"평가 세트 {len(rows)}문항 / live 호출 대상 {len(all_targets)}건 (반복 {args.repeat}회, 모델 {model})")
    print(f"완료 {len(all_targets) - len(pending_all)}건, 남음 {len(pending_all)}건 (결과 파일 {n_files}개 병합)")
    for group, (done, total) in sorted(per_group.items()):
        print(f"  그룹 {group}: {done}/{total}")
    if stats["stale"] or stats["other_model"]:
        print(f"  (제외: 라벨·질문 변경 {stats['stale']}건, 다른 모델 {stats['other_model']}건)")

    if not (args.run or args.merge_only):
        print("\n--run 으로 이어서 실행하거나 --merge-only 로 보고서를 만드세요. (지금은 현황만 출력했습니다)")
        return 0

    stopped = None
    if args.run:
        limit = pending_run if args.budget <= 0 else pending_run[:args.budget]
        if not limit:
            print("\n이번에 실행할 남은 문항이 없습니다.")
        else:
            minutes = len(limit) * args.sleep / 60
            print(f"\n이번 실행: {len(limit)}회 호출, 호출 간 {args.sleep:g}초 (약 {minutes:.0f}분)")
            try:
                _, stopped = run_pending(pending_run, catalog, out_dir, args.budget, args.sleep, model)
            except RuntimeError as exc:  # API 키 없음
                print(f"[오류] {exc}")
                return 2
            results, stats, n_files = load_results(out_dir, rows, catalog, model)
            per_group, pending_all = progress(all_targets, results)
            if stopped:
                print(f"  ⚠ {stopped}")

    rules = run_eval.summarize_rules(run_eval.evaluate_rules(rows, catalog))
    report = run_eval.build_report(rows, rules, results, stopped, args)
    block = progress_block(per_group, pending_all, stats, n_files, model)
    report = report.replace("\n## 1. 규칙 계층", block + "\n## 1. 규칙 계층", 1)
    md_path, csv_path = write_merged(report, results, out_dir)
    print(f"\n합산 보고서: {md_path}\n건별 합산: {csv_path}")
    violations = run_eval.summarize_live(results)["violations"] if results else 0
    return 0 if (rules["passed"] == rules["total"] and violations == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
