"""run_eval_resume.py 오프라인 점검 (API 호출 없음 · 임시 폴더만 사용 · 실제 output/eval 은 건드리지 않음)

사용 (프로젝트 루트에서):
    python -m unittest src/test_eval_resume.py -v
"""
import contextlib
import csv
import io
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_eval  # noqa: E402
import run_eval_resume as rer  # noqa: E402

COLUMNS = run_eval.COLUMNS
CATALOG = {"8507600000": {}, "8806210000": {}, "8504400000": {}}
LEGACY_FIELDS = ["kind", "id", "group", "difficulty", "trial", "query", "expected", "predicted", "confidence", "outcome", "note"]
NEW_FIELDS = LEGACY_FIELDS + ["blocked_event", "violation", "model", "timestamp"]


def make_row(row_id, group, query, route="", hsk="", **extra):
    row = {c: "" for c in COLUMNS}
    row.update(id=row_id, group=group, query=query, expected_route=route, expected_hsk=hsk)
    row.update(extra)
    return row


def fixture_rows():
    return [
        make_row("D01", "D_범위밖", "Q-D01 프랑스산 와인 10병", route="customs", hsk="NONE"),
        make_row("D02", "D_범위밖", "Q-D02 3D 프린터 2대", hsk="NONE"),
        make_row("G01", "G_보안", "Q-G01 연구용 리튬이온 배터리 5개", hsk="8507600000"),
        make_row("C01", "C_혼동", "Q-C01 드론에 장착할 배터리 30개", hsk="8507600000"),
        make_row("A01", "A_기본", "Q-A01 연구개발용 배터리 25개", route="customs", hsk="8507600000"),
        make_row("E01", "E_의도", "Q-E01 연구실 소화기 점검 주기", route="lab_safety"),
        make_row("F01", "F_규칙", "Q-F01 규칙만 보는 문항", hint_hsk="8507600000", expected_status="success", expected_stage="done"),
    ]  # live 호출 대상: D01 route, D01 hsk, D02 hsk, G01 hsk, C01 hsk, A01 route, A01 hsk, E01 route = 8건


class FakeRoute:
    def __init__(self, category):
        self.category, self.confidence, self.reason, self.fallback_used = category, 0.9, "ok", False


class Fakes:
    """가짜 라우터·추론기. events 에 (종류, 질문 앞부분) 을 기록한다."""

    def __init__(self):
        self.events = []
        self.hsk_override = {}  # 질문 id → infer 결과 dict

    def route(self, query):
        self.events.append(("route", query.split()[0]))
        return FakeRoute("lab_safety" if "소화기" in query else "customs_and_certification")

    def infer(self, query):
        tag = query.split()[0]
        self.events.append(("hsk", tag))
        for key, result in self.hsk_override.items():
            if key in tag:
                return result
        if "와인" in query or "프린터" in query:
            return {"status": "error", "message": "지원 대상 품목을 특정하지 못했습니다. (사유: 목록 밖)"}
        return {"status": "success", "data": {"hsk_code": "8507600000", "confidence": 0.9}}


QUOTA = {"status": "error", "error_kind": "quota", "message": "API 사용량 한도를 초과했습니다."}
OVERLOADED = {"status": "error", "error_kind": "overloaded", "message": "AI 서버가 혼잡합니다."}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.rows = fixture_rows()
        self.fakes = Fakes()
        self._saved = (run_eval.EVAL_OUT_DIR, run_eval.config.MODEL_NAME, run_eval.evaluate_rules,
                       rer._load_catalog, rer._load_rows, rer._get_clients, rer.time.sleep)
        run_eval.EVAL_OUT_DIR = self.out
        run_eval.config.MODEL_NAME = "m1"
        run_eval.evaluate_rules = lambda rows, catalog: []   # 규칙 게이트(오프라인)는 여기서 검증하지 않음
        rer._load_catalog = lambda: CATALOG
        rer._load_rows = lambda: self.rows
        rer._get_clients = lambda need_router=True: (self.fakes, self.fakes.infer)
        rer.time.sleep = lambda s: None

    def tearDown(self):
        (run_eval.EVAL_OUT_DIR, run_eval.config.MODEL_NAME, run_eval.evaluate_rules,
         rer._load_catalog, rer._load_rows, rer._get_clients, rer.time.sleep) = self._saved
        self.tmp.cleanup()

    def run_main(self, *argv):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = rer.main(list(argv))
        return code, buffer.getvalue()

    def write_csv(self, name, rows, fields):
        path = self.out / name
        with open(path, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return path

    def result_row(self, kind, row_id, outcome, **extra):
        by_id = {r["id"]: r for r in self.rows}
        base = by_id[row_id]
        expected = base["expected_route"] if kind == "route" else base["expected_hsk"]
        row = {"kind": kind, "id": row_id, "group": base["group"], "difficulty": "", "trial": 1,
               "query": base["query"], "expected": expected, "predicted": "", "confidence": "",
               "outcome": outcome, "note": "", "model": "m1"}
        row.update(extra)
        return row


class TargetTests(Base):
    def test_order_and_targets(self):
        self.assertEqual(rer.resolve_order("", self.rows), ["D", "G", "C", "A", "B", "E", "F"])
        self.assertEqual(rer.resolve_order("c, a", self.rows), ["C", "A"])
        targets = rer.build_targets(self.rows, rer.resolve_order("", self.rows), 1)
        self.assertEqual([(k, r["id"]) for k, r, _ in targets],
                         [("route", "D01"), ("hsk", "D01"), ("hsk", "D02"), ("hsk", "G01"), ("hsk", "C01"),
                          ("route", "A01"), ("hsk", "A01"), ("route", "E01")])
        self.assertEqual(len(rer.build_targets(self.rows, ["D", "G", "C", "A", "E"], 3)), 24)


class RunTests(Base):
    def test_budget_and_resume_without_duplicates(self):
        code, _ = self.run_main("--run", "--budget", "4", "--sleep", "0")
        self.assertEqual(code, 0)
        self.assertEqual(self.fakes.events, [("route", "Q-D01"), ("hsk", "Q-D01"), ("hsk", "Q-D02"), ("hsk", "Q-G01")])
        self.assertEqual(len(list(self.out.glob("eval_results_*_resume.csv"))), 1)

        self.run_main("--run", "--budget", "4", "--sleep", "0")
        self.assertEqual(len(self.fakes.events), 8)
        self.assertEqual(len(set(self.fakes.events)), 8)  # 앞서 한 문항을 다시 부르지 않음

        _, text = self.run_main("--run", "--budget", "4", "--sleep", "0")
        self.assertEqual(len(self.fakes.events), 8)
        self.assertIn("남은 문항이 없습니다", text)

        # 합산 결과물은 다음 합산에서 다시 읽히지 않는 이름이어야 한다
        names = {p.name for p in self.out.glob("eval_results_*.csv")}
        self.assertTrue(all(n.endswith("_resume.csv") for n in names), names)
        self.assertGreaterEqual(len(list(self.out.glob("eval_report_merged_*.md"))), 1)
        self.assertGreaterEqual(len(list(self.out.glob("eval_merged_*.csv"))), 1)

    def test_progress_saved_after_each_call(self):
        self.run_main("--run", "--budget", "3", "--sleep", "0")
        files = list(self.out.glob("eval_results_*_resume.csv"))
        with open(files[0], encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([r["id"] for r in rows], ["D01", "D01", "D02"])
        self.assertTrue(all(r["model"] == "m1" and r["timestamp"] for r in rows))

    def test_quota_stops_and_item_is_retried(self):
        self.fakes.hsk_override["G01"] = QUOTA
        code, text = self.run_main("--run", "--budget", "8", "--sleep", "0")
        self.assertEqual(code, 0)
        self.assertEqual(len(self.fakes.events), 4)            # D01 route/hsk, D02, G01(429) 에서 멈춤
        self.assertIn("429", text)

        self.fakes.hsk_override.clear()
        self.run_main("--run", "--budget", "8", "--sleep", "0")
        self.assertEqual(self.fakes.events[4:], [("hsk", "Q-G01"), ("hsk", "Q-C01"), ("route", "Q-A01"),
                                                 ("hsk", "Q-A01"), ("route", "Q-E01")])
        results, _, _ = rer.load_results(self.out, self.rows, CATALOG, "m1")
        g01 = [r for r in results if r["id"] == "G01"]
        self.assertEqual(len(g01), 1)
        self.assertNotEqual(g01[0]["outcome"], "api_error")  # 재실행 결과가 유효 결과로 대체됨

    def test_three_consecutive_api_errors_stop(self):
        for tag in ("D01", "D02", "G01", "C01"):
            self.fakes.hsk_override[tag] = OVERLOADED
        _, text = self.run_main("--run", "--budget", "0", "--sleep", "0")
        self.assertEqual(len(self.fakes.events), 4)            # route 성공 1 + 연속 오류 3
        self.assertIn("연속", text)

    def test_groups_filter_and_order(self):
        self.run_main("--run", "--groups", "C,A", "--budget", "10", "--sleep", "0")
        self.assertEqual(self.fakes.events, [("hsk", "Q-C01"), ("route", "Q-A01"), ("hsk", "Q-A01")])
        _, status = self.run_main("--status")
        self.assertIn("남음 5건", status)                      # D·G·E 는 이번에 지정하지 않아 그대로 남음

    def test_repeat_makes_new_keys(self):
        self.run_main("--run", "--repeat", "2", "--budget", "0", "--sleep", "0")
        self.assertEqual(len(self.fakes.events), 16)
        self.run_main("--run", "--repeat", "1", "--budget", "0", "--sleep", "0")
        self.assertEqual(len(self.fakes.events), 16)           # 1회차 결과는 이미 있으므로 추가 호출 없음

    def test_no_key_returns_2_and_writes_nothing(self):
        def no_key(need_router=True):
            raise RuntimeError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
        rer._get_clients = no_key
        code, text = self.run_main("--run", "--budget", "4", "--sleep", "0")
        self.assertEqual(code, 2)
        self.assertIn("GEMINI_API_KEY", text)
        self.assertEqual(list(self.out.iterdir()), [])


class MergeTests(Base):
    def test_status_and_merge_only_make_no_api_calls(self):
        def forbidden(need_router=True):
            raise AssertionError("API 클라이언트를 만들면 안 됩니다")
        rer._get_clients = forbidden

        code, text = self.run_main("--status")
        self.assertEqual(code, 0)
        self.assertEqual(list(self.out.iterdir()), [])
        code, text = self.run_main()
        self.assertEqual(code, 0)
        self.assertIn("현황만 출력", text)

        code, text = self.run_main("--merge-only")
        self.assertEqual(code, 0)
        report = next(self.out.glob("eval_report_merged_*.md")).read_text(encoding="utf-8")
        self.assertIn("## 0. 누적 진행 현황", report)
        self.assertIn("완료 호출: 0 / 8", report)
        self.assertIn("## 1. 규칙 계층", report)

    def test_legacy_csv_flags_are_reconstructed(self):
        rows = [
            self.result_row("hsk", "G01", "wrong", predicted="9999999999", confidence="0.90", model=""),
            self.result_row("hsk", "C01", "blocked", note="품목표에 없는 HSK 코드가 추론되어 차단했습니다: 1234", model=""),
            self.result_row("hsk", "D02", "correct_reject", note="품목표에 없는 HSK 코드가 추론되어 차단했습니다: 5678", model=""),
            self.result_row("hsk", "D01", "api_error", note="quota: API 사용량 한도를 초과했습니다.", model=""),
            self.result_row("route", "A01", "correct", predicted="customs_and_certification", confidence="0.90", model=""),
            {"kind": "rule/수량 추출", "id": "A01", "group": "A_기본", "query": "x", "expected": "1", "predicted": "1", "outcome": "ok"},
        ]
        self.write_csv("eval_results_20260101_000000.csv", rows, LEGACY_FIELDS)  # run_eval.py 가 만든 형식(model 열 없음)

        results, stats, n_files = rer.load_results(self.out, self.rows, CATALOG, "m1")
        self.assertEqual((len(results), n_files), (5, 1))            # rule 줄은 읽지 않음
        self.assertEqual(stats["unknown_model"], 4)                  # 유효한 4건은 모델 기록 없음
        summary = run_eval.summarize_live(results)
        self.assertEqual(summary["violations"], 1)
        self.assertEqual(summary["blocked_events"], 2)
        self.assertEqual(summary["hsk_excluded"], 1)
        self.assertEqual(summary["in_scope_blocked"], 1)

        rules = run_eval.summarize_rules([])
        args = type("A", (), {"repeat": 1, "sleep": 15})()
        report = run_eval.build_report(self.rows, rules, results, None, args)
        self.assertIn("목록 외 코드가 결과로 나온 건수", report)

    def test_stale_and_other_model_are_excluded(self):
        rows = [
            self.result_row("hsk", "G01", "correct", predicted="8507600000", model="other"),     # 다른 모델
            self.result_row("hsk", "C01", "correct", predicted="8507600000", query="바뀌기 전 질문"),  # 질문 변경
            self.result_row("hsk", "A01", "correct", predicted="8806210000", expected="8806210000"),  # 라벨 변경
            self.result_row("hsk", "D02", "correct_reject", model=""),                            # 기록 없음 → 허용
            self.result_row("route", "E01", "correct", model="m1"),                               # 현재 모델
            self.result_row("hsk", "ZZZ", "correct"),                                             # 없는 id
        ]
        self.rows_ref = rows
        self.write_csv("eval_results_20260101_000000.csv", [dict(r, blocked_event=0, violation=0) for r in rows], NEW_FIELDS)
        results, stats, _ = rer.load_results(self.out, self.rows, CATALOG, "m1")
        self.assertEqual(sorted(r["id"] for r in results), ["D02", "E01"])
        self.assertEqual((stats["other_model"], stats["stale"], stats["unknown_model"]), (1, 3, 1))

    def test_newest_valid_result_wins_and_errors_never_overwrite(self):
        older = [
            self.result_row("hsk", "G01", "correct"),                   # 뒤 파일의 api_error 가 덮어쓰면 안 됨
            self.result_row("hsk", "C01", "api_error", note="overloaded: x"),   # 뒤 파일의 유효 결과가 대체
            self.result_row("hsk", "D02", "wrong"),                     # 뒤 파일의 유효 결과가 이김
        ]
        newer = [
            self.result_row("hsk", "G01", "api_error", note="quota: x"),
            self.result_row("hsk", "C01", "correct"),
            self.result_row("hsk", "D02", "correct_reject"),
        ]
        self.write_csv("eval_results_20260101_000000.csv", older, NEW_FIELDS)
        self.write_csv("eval_results_20260102_000000.csv", newer, NEW_FIELDS)
        results, _, n_files = rer.load_results(self.out, self.rows, CATALOG, "m1")
        by_id = {r["id"]: r["outcome"] for r in results}
        self.assertEqual(n_files, 2)
        self.assertEqual(by_id, {"G01": "correct", "C01": "correct", "D02": "correct_reject"})

    def test_progress_counts_only_valid_results(self):
        rows = [self.result_row("hsk", "G01", "correct"), self.result_row("hsk", "C01", "api_error", note="quota: x")]
        self.write_csv("eval_results_20260101_000000.csv", rows, NEW_FIELDS)
        results, _, _ = rer.load_results(self.out, self.rows, CATALOG, "m1")
        targets = rer.build_targets(self.rows, rer.resolve_order("", self.rows), 1)
        per_group, pending = rer.progress(targets, results)
        self.assertEqual(per_group["G"], [1, 1])
        self.assertEqual(per_group["C"], [0, 1])                        # api_error 는 완료가 아님
        self.assertIn(("hsk", "C01"), [(k, r["id"]) for k, r, _ in pending])
        self.assertEqual(len(pending), 7)

    def test_report_after_full_run_has_progress_block(self):
        self.run_main("--run", "--budget", "0", "--sleep", "0")
        report = next(self.out.glob("eval_report_merged_*.md")).read_text(encoding="utf-8")
        self.assertIn("완료 호출: 8 / 8 (100%)", report)
        self.assertIn("모든 대상 문항이 완료되었습니다", report)
        self.assertIn("품목 정확도", report)
        merged = next(self.out.glob("eval_merged_*.csv"))
        with open(merged, encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        self.assertEqual(reader.fieldnames, rer.FIELDS)
        self.assertEqual(len(rows), 8)
        self.assertEqual(Counter(r["kind"] for r in rows), Counter({"hsk": 5, "route": 3}))

    def test_partial_report_warns_about_incomplete_numbers(self):
        self.run_main("--run", "--budget", "3", "--sleep", "0")
        report = next(self.out.glob("eval_report_merged_*.md")).read_text(encoding="utf-8")
        self.assertIn("아직 안 한 항목 5건", report)
        self.assertIn("문항 수(n)", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
