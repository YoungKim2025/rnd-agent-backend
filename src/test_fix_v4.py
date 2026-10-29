"""rag_store / pipeline_runner / app 수정본 오프라인 점검 (API 호출 없음 · 임시 폴더와 가짜 객체만 사용)

사용 (프로젝트 루트에서):
    python src/test_fix_v4.py
"""
import contextlib
import json
import runpy
import shutil
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

if hasattr(sys.stdout, "reconfigure"):  # Windows 콘솔(cp949) 대비
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import rag_store  # noqa: E402
import pipeline_runner as pr  # noqa: E402

APP_SOURCE = Path(__file__).resolve().parent / "app.py"
DEFAULT_QUERY = (
    "연구개발 목적으로 리튬이온 배터리 프로토타입 100개를 수입하려고 해. "
    "개당 가격은 5천원이고 인증은 아직 없어. 전안법 면제 조건과 관세율 알려줘."
)


# ═════════════════════════ rag_store ═════════════════════════
class RagStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_meta(self, **meta):
        (self.tmp / rag_store.META_NAME).write_text(json.dumps(meta), encoding="utf-8")

    def test_current_db_passes_without_embedding_signature(self):
        # 실제 DB 서명. 예전 코드는 'models/embedding-001' 로 고정 대조해서 이 DB를 항상 거부했다.
        self.write_meta(backend="gemini", model="gemini-embedding-001", schema=rag_store.SCHEMA_VERSION)
        rag_store.check_store_meta(chroma_dir=self.tmp)

    def test_embedding_signature_is_checked_only_when_requested(self):
        self.write_meta(backend="gemini", model="gemini-embedding-001", schema=rag_store.SCHEMA_VERSION)
        rag_store.check_store_meta({"backend": "gemini", "model": "gemini-embedding-001"}, self.tmp)
        with self.assertRaises(rag_store.VectorStoreError):
            rag_store.check_store_meta({"backend": "gemini", "model": "models/embedding-001"}, self.tmp)

    def test_schema_mismatch_missing_and_corrupt_files_are_rejected(self):
        with self.assertRaises(rag_store.VectorStoreError):  # 파일 없음
            rag_store.check_store_meta(chroma_dir=self.tmp)
        self.write_meta(backend="gemini", model="x", schema=rag_store.SCHEMA_VERSION + 1)
        with self.assertRaises(rag_store.VectorStoreError):  # 구조 버전 불일치
            rag_store.check_store_meta(chroma_dir=self.tmp)
        (self.tmp / rag_store.META_NAME).write_text("{깨진 json", encoding="utf-8")
        with self.assertRaises(rag_store.VectorStoreError):  # 읽을 수 없음
            rag_store.check_store_meta(chroma_dir=self.tmp)

    def test_load_vectorstore_accepts_the_real_db_signature(self):
        self.write_meta(backend="gemini", model="gemini-embedding-001", schema=rag_store.SCHEMA_VERSION)
        fake_module = types.ModuleType("langchain_chroma")
        fake_module.Chroma = lambda **kwargs: {"opened": kwargs["persist_directory"]}
        with mock.patch.dict(sys.modules, {"langchain_chroma": fake_module}), \
                mock.patch.object(rag_store, "CHROMA_DIR", self.tmp):
            store = rag_store.load_vectorstore()
        self.assertEqual(store, {"opened": str(self.tmp)})

    def test_retrieve_uses_metadata_filters_specific_first(self):
        calls = []

        class FakeStore:
            def get(self, where):
                calls.append(where)
                if any(rag_store.hsk6_flag("8507600000") in cond for cond in where["$and"]):
                    return {"documents": ["전용 조항"], "metadatas": [{"section": 2, "part": 0, "title": "2. 배터리"}]}
                return {"documents": ["일반 조항", "중복 키"],
                        "metadatas": [{"section": 7, "part": 0}, {"section": 2, "part": 0}]}

        docs = rag_store.retrieve_legal_docs("무관한 질문", "8507600000", "전안법", store=FakeStore())
        self.assertEqual([d.page_content for d in docs], ["전용 조항", "일반 조항"])  # 같은 (section, part) 는 한 번만
        self.assertEqual(len(calls), 2)
        general = calls[1]["$and"]
        self.assertIn({"scope": "general"}, general)
        self.assertIn({"law_type": {"$in": ["common", "complex", "electrical"]}}, general)


# ═════════════════════════ pipeline_runner: 캐시 ═════════════════════════
class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self._patches = [mock.patch.object(pr, "CACHE_DIR", self.tmp), mock.patch.object(pr, "_hash_targets", lambda: [])]
        for patcher in self._patches:
            patcher.start()

    def tearDown(self):
        for patcher in self._patches:
            patcher.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_hash_ignores_line_endings_but_not_content(self):
        crlf, lf, other = self.tmp / "a.txt", self.tmp / "b.txt", self.tmp / "c.txt"
        crlf.write_bytes("첫 줄\r\n둘째 줄\r\n".encode("utf-8"))
        lf.write_bytes("첫 줄\n둘째 줄\n".encode("utf-8"))
        other.write_bytes("첫 줄\n다른 줄\n".encode("utf-8"))
        self.assertEqual(pr._normalized_hash([crlf]), pr._normalized_hash([lf]))   # Windows 와 Linux 에서 같은 키
        self.assertNotEqual(pr._normalized_hash([lf]), pr._normalized_hash([other]))

    def test_key_depends_on_inputs_but_not_on_whitespace(self):
        base = pr._cache_key("배터리 10개 수입", "m1", True)
        self.assertEqual(base, pr._cache_key("  배터리   10개 수입  ", "m1", True))
        self.assertNotEqual(base, pr._cache_key("배터리 11개 수입", "m1", True))
        self.assertNotEqual(base, pr._cache_key("배터리 10개 수입", "m2", True))
        self.assertNotEqual(base, pr._cache_key("배터리 10개 수입", "m1", False))

    def test_only_success_is_saved_and_loaded(self):
        path = pr._cache_path("k1")
        pr._save_cache(path, {"status": "error", "message": "429"})
        self.assertFalse(path.exists())                       # 실패는 저장하지 않는다
        pr._save_cache(path, {"status": "success", "report_markdown": "# r"})
        loaded = pr._load_cache(path)
        self.assertEqual((loaded["status"], loaded["cache_hit"]), ("success", True))
        self.assertEqual([p.name for p in self.tmp.glob("*.tmp")], [])   # 임시 파일이 남지 않는다
        self.assertTrue(path.name.startswith("cache_v3_"))

    def test_failed_or_corrupt_cache_files_are_deleted(self):
        bad, broken = pr._cache_path("bad"), pr._cache_path("broken")
        bad.write_text(json.dumps({"status": "error", "message": "예전에 저장된 429"}), encoding="utf-8")
        broken.write_text("{깨진", encoding="utf-8")
        self.assertIsNone(pr._load_cache(bad))
        self.assertIsNone(pr._load_cache(broken))
        self.assertFalse(bad.exists() or broken.exists())

    def test_clear_cache_removes_every_version(self):
        (self.tmp / "cache_v3_a.json").write_text("{}", encoding="utf-8")
        (self.tmp / "cache_e0f3814a121ab458421f3f5e4d6384ea.json").write_text("{}", encoding="utf-8")  # 예전 형식
        (self.tmp / "keep.txt").write_text("남아야 함", encoding="utf-8")
        self.assertEqual(pr.clear_cache(), 2)
        self.assertTrue((self.tmp / "keep.txt").exists())


# ═════════════════════════ pipeline_runner: 템플릿 모드 ═════════════════════════
class TemplateModeTests(unittest.TestCase):
    def setUp(self):
        self.dispatcher = pr._DISPATCHER
        self._real = self.dispatcher._rnd_real
        self.dispatcher._rnd_real = lambda prompt: "REAL"

    def tearDown(self):
        self.dispatcher._rnd_real = self._real

    def test_dispatcher_is_installed_once(self):
        self.assertTrue(getattr(config.generate_text, "_rnd_dispatcher", False))
        self.assertIs(pr._install_template_dispatcher(), config.generate_text)   # 다시 불러도 중복으로 감싸지 않는다

    def test_flag_is_thread_local(self):
        seen = {}

        def other_thread():
            seen["other"] = config.generate_text("p")   # 다른 접속자의 요청: 평소처럼 실제 함수

        with pr._template_mode(True):
            with self.assertRaises(pr.TemplateModeSkip):
                config.generate_text("p")                # 템플릿 모드인 이 스레드만 건너뜀
            thread = threading.Thread(target=other_thread)
            thread.start()
            thread.join()
        self.assertEqual(seen["other"], "REAL")
        self.assertEqual(config.generate_text("p"), "REAL")   # 블록을 나오면 원래대로

    def test_exception_inside_block_restores_everything(self):
        with self.assertRaises(RuntimeError):
            with pr._template_mode(True):
                raise RuntimeError("보고서 생성 중 예외")
        self.assertEqual(config.generate_text("p"), "REAL")   # 예전 코드는 여기서 가짜 함수가 영구히 남았다


# ═════════════════════════ pipeline_runner: 흐름 ═════════════════════════
class FakeRoute:
    def __init__(self, category, fallback_used=False, reason="ok"):
        self.category, self.confidence, self.reason, self.fallback_used = category, 0.9, reason, fallback_used


class FakeRouter:
    def __init__(self, owner):
        self.owner = owner

    def route(self, text):
        self.owner.calls["route"] += 1
        return self.owner.route_result


class FlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.calls = {"route": 0, "infer": 0, "report": 0}
        self.template_flags = []
        self.route_result = FakeRoute(pr.CUSTOMS)
        self.infer_result = {"status": "success", "data": {
            "hsk_code": "8507600000", "item_name": "리튬이온 축전지 및 배터리 팩", "confidence": 0.95, "reason": "ok"}}
        self.report_error = None

        def fake_infer(text):
            self.calls["infer"] += 1
            return self.infer_result

        def fake_report(text, hsk_hint):
            self.calls["report"] += 1
            self.template_flags.append(getattr(pr._DISPATCHER._rnd_local, "template", False))
            if self.report_error:
                raise self.report_error
            return {"status": "success", "report_markdown": f"# 보고서 {hsk_hint}"}

        self._patches = [
            mock.patch.object(pr, "CACHE_DIR", self.tmp),
            mock.patch.object(pr, "_hash_targets", lambda: []),
            mock.patch.object(pr, "infer_hsk_code", fake_infer),
            mock.patch.object(pr, "generate_fast_track_report", fake_report),
            mock.patch.object(pr, "_get_router", lambda: FakeRouter(self)),
        ]
        for patcher in self._patches:
            patcher.start()

    def tearDown(self):
        for patcher in self._patches:
            patcher.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_default_query_uses_rule_route_then_hsk_then_report(self):
        result = pr.run_pipeline(DEFAULT_QUERY)
        self.assertEqual((result["status"], result["stage"]), ("success", "done"))
        self.assertEqual(self.calls, {"route": 0, "infer": 1, "report": 1})   # 라우터(LLM) 호출 없음
        self.assertEqual(result["route"]["reason"], "규칙 기반 식별(수입 관련 표현 + 수량 확인됨)")
        self.assertEqual(self.template_flags, [True])                           # 템플릿 모드는 보고서 단계에서만 켜진다

    def test_template_flag_off_when_disabled(self):
        pr.run_pipeline(DEFAULT_QUERY, use_template=False)
        self.assertEqual(self.template_flags, [False])

    def test_second_identical_request_is_served_from_cache(self):
        pr.run_pipeline(DEFAULT_QUERY)
        again = pr.run_pipeline("  " + DEFAULT_QUERY.replace(" ", "  ") + " ")   # 공백만 다른 같은 질문
        self.assertTrue(again["cache_hit"])
        self.assertEqual(self.calls["infer"], 1)

    def test_lab_safety_question_without_import_terms_reaches_the_router(self):
        self.route_result = FakeRoute(pr.LAB_SAFETY)
        result = pr.run_pipeline("연구실 소화기 3대 점검 주기를 알려줘")     # '연구' + 수량이 있어도 통관으로 오분류하지 않는다
        self.assertEqual((result["status"], result["stage"]), ("unsupported", "router"))
        self.assertEqual(self.calls, {"route": 1, "infer": 0, "report": 0})

    def test_greeting_without_quantity_is_not_an_input_error(self):
        self.route_result = FakeRoute("general_chat")
        result = pr.run_pipeline("안녕하세요")
        self.assertEqual((result["status"], result["stage"]), ("unsupported", "router"))   # 예전: '수량을 파악할 수 없습니다'
        self.assertEqual(self.calls["route"], 1)

    def test_customs_question_without_quantity_asks_for_quantity(self):
        result = pr.run_pipeline("통관 서류가 뭐야?")        # 라우터가 통관으로 판단 → 그다음 수량 검사
        self.assertEqual((result["status"], result["stage"]), ("error", "report"))
        self.assertIn("수량", result["message"])
        self.assertEqual(self.calls["infer"], 0)

    def test_commercial_purpose_is_blocked_before_any_llm_call(self):
        result = pr.run_pipeline("판매할 목적으로 태블릿 PC 50대를 수입하려고 해.")
        self.assertEqual((result["status"], result["stage"]), ("error", "report"))
        self.assertEqual(self.calls, {"route": 0, "infer": 0, "report": 0})

    def test_foreign_price_is_blocked_before_any_llm_call(self):
        result = pr.run_pipeline("연구용 배터리 5개를 개당 $10 에 수입하려고 해")
        self.assertEqual((result["status"], result["stage"]), ("error", "report"))
        self.assertIn("외화", result["message"])
        self.assertEqual(self.calls, {"route": 0, "infer": 0, "report": 0})

    def test_llm_outage_is_not_reported_as_an_unrelated_question(self):
        self.route_result = FakeRoute(
            pr.UNKNOWN, fallback_used=True,
            reason="규칙 기반 대체 발동: API 사용량 한도를 초과했습니다. 잠시 후 다시 시도해 주세요.")
        result = pr.run_pipeline("안녕하세요")
        self.assertEqual((result["status"], result["stage"], result["error_kind"]), ("error", "router", "quota"))
        self.assertIn("사용량 한도", result["message"])
        self.assertNotIn("관련된 질문이 아닙니다", result["message"])

    def test_hsk_api_error_keeps_error_kind_and_is_never_cached(self):
        self.infer_result = {"status": "error", "error_kind": "quota", "message": "API 사용량 한도를 초과했습니다."}
        first = pr.run_pipeline(DEFAULT_QUERY)
        self.assertEqual((first["stage"], first["error_kind"]), ("hsk", "quota"))
        self.assertEqual(list(self.tmp.glob("cache_*.json")), [])
        self.infer_result = {"status": "success", "data": {"hsk_code": "8507600000", "item_name": "배터리", "confidence": 1.0}}
        second = pr.run_pipeline(DEFAULT_QUERY)               # 한도가 풀린 뒤 같은 질문은 실제로 다시 시도된다
        self.assertEqual(second["status"], "success")
        self.assertEqual(self.calls["infer"], 2)

    def test_unsupported_item_stops_at_hsk(self):
        self.infer_result = {"status": "success", "data": {"hsk_code": "", "reason": "목록 밖"}}
        result = pr.run_pipeline("프랑스산 와인 10개를 수입하려고 해")
        self.assertEqual((result["status"], result["stage"]), ("error", "hsk"))
        self.assertEqual(self.calls["report"], 0)

    def test_report_exception_is_contained_and_flag_is_restored(self):
        self.report_error = RuntimeError("보고서 생성 실패")
        result = pr.run_pipeline(DEFAULT_QUERY)
        self.assertEqual((result["status"], result["stage"]), ("error", "report"))
        self.assertFalse(getattr(pr._DISPATCHER._rnd_local, "template", False))
        self.assertEqual(list(self.tmp.glob("cache_*.json")), [])


# ═════════════════════════ app.py ═════════════════════════
class SessionState(dict):
    """st.session_state 흉내: dict + 속성 접근."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class AppHarness:
    """app.py 를 임시 프로젝트 구조(<tmp>/proj/src/app.py)에 복사해 가짜 streamlit 으로 실행한다."""

    def __init__(self, test_case):
        self.tmp = Path(tempfile.mkdtemp())
        test_case.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.proj = self.tmp / "proj"
        (self.proj / "src").mkdir(parents=True)
        (self.proj / "demo").mkdir()
        (self.proj / "cache").mkdir()
        shutil.copy(APP_SOURCE, self.proj / "src" / "app.py")
        demo = {"query": DEFAULT_QUERY, "generated_at": "테스트 시각",
                "result": {"status": "success", "report_markdown": "# 시연 보고서", "warnings": []}}
        (self.proj / "demo" / "demo_default.json").write_text(json.dumps(demo, ensure_ascii=False), encoding="utf-8")
        self.run_calls = []
        self.run_result = {"status": "success", "report_markdown": "# 실시간 보고서"}

    def run(self, query=DEFAULT_QUERY, submit=False, buttons=None, template=True, demo=True, query_params=None):
        st = mock.MagicMock(name="streamlit")
        st.session_state = SessionState()
        st.query_params = dict(query_params or {})
        self.columns_created = []

        def make_columns(spec):
            cols = [mock.MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))]
            self.columns_created.extend(cols)
            return cols

        st.columns.side_effect = make_columns
        st.text_area.side_effect = lambda label, value="", **kwargs: query
        st.form_submit_button.return_value = submit
        st.button.side_effect = lambda label, **kwargs: bool((buttons or {}).get(label, False))
        st.sidebar.checkbox.side_effect = lambda label, value=False, **kwargs: demo if "시연" in label else template

        fake_pipeline = types.ModuleType("pipeline_runner")
        fake_pipeline.run_pipeline = lambda q, use_template=True: (self.run_calls.append((q, use_template)) or self.run_result)
        with mock.patch.dict(sys.modules, {"streamlit": st, "pipeline_runner": fake_pipeline}):
            namespace = runpy.run_path(str(self.proj / "src" / "app.py"), run_name="__main__")
        self.st, self.namespace = st, namespace
        return st

    @staticmethod
    def texts(mock_method):
        return [str(call.args[0]) if call.args else "" for call in mock_method.call_args_list]


class AppTests(unittest.TestCase):
    def setUp(self):
        self.app = AppHarness(self)

    def test_first_streamlit_call_is_set_page_config_and_default_query_matches(self):
        st = self.app.run()
        self.assertEqual(st.method_calls[0][0], "set_page_config")
        self.assertEqual(self.app.namespace["DEFAULT_QUERY"], DEFAULT_QUERY)
        self.assertEqual(self.app.run_calls, [])             # 버튼을 누르기 전에는 분석하지 않는다

    def test_exact_default_query_shows_labeled_demo_result(self):
        st = self.app.run(submit=True)
        self.assertEqual(self.app.run_calls, [])             # API 호출 없음
        self.assertTrue(st.session_state["result"]["demo"])
        self.assertTrue(any("저장된 시연 결과" in t and "실시간 분석이 아닙니다" in t for t in self.app.texts(st.info)))
        self.assertFalse(any("분석 및 문서 생성이 완료" in t for t in self.app.texts(st.success)))

    def test_whitespace_variants_of_default_query_still_match_exactly(self):
        self.app.run(query="  " + DEFAULT_QUERY.replace(" ", "   "), submit=True)
        self.assertEqual(self.app.run_calls, [])

    def test_query_that_only_starts_like_the_default_runs_live(self):
        query = "연구개발 목적으로 리튬이온 배터리 5개를 수입하려고 해."   # 예전 코드는 이것도 100개짜리 옛 결과를 보여 줬다
        st = self.app.run(query=query, submit=True)
        self.assertEqual(self.app.run_calls, [(query, True)])
        self.assertNotIn("demo", st.session_state["result"])

    def test_demo_checkbox_off_runs_default_query_live(self):
        self.app.run(submit=True, demo=False, template=False)
        self.assertEqual(self.app.run_calls, [(DEFAULT_QUERY, False)])

    def test_empty_query_warns_and_does_not_run(self):
        st = self.app.run(query="   ", submit=True)
        self.assertEqual(self.app.run_calls, [])
        self.assertTrue(any("질문을 입력" in t for t in self.app.texts(st.warning)))

    def test_cache_reset_clears_the_real_cache_folder_and_nothing_else(self):
        cache = self.app.proj / "cache"
        (cache / "cache_v3_a.json").write_text("{}", encoding="utf-8")
        (cache / "cache_old.json").write_text("{}", encoding="utf-8")
        (cache / "keep.txt").write_text("x", encoding="utf-8")
        wrong = self.app.tmp / "cache"                       # 예전 코드가 가리키던 상위 폴더
        wrong.mkdir()
        (wrong / "other.json").write_text("{}", encoding="utf-8")

        st = self.app.run(buttons={"🧹 캐시 초기화": True})
        self.assertEqual(sorted(p.name for p in cache.iterdir()), ["keep.txt"])
        self.assertTrue((wrong / "other.json").exists())
        self.assertTrue(any("2건" in t for t in self.app.texts(st.success)))

    def test_transient_errors_are_warnings_other_errors_are_errors(self):
        self.app.run_result = {"status": "error", "stage": "hsk", "error_kind": "quota", "message": "API 사용량 한도를 초과했습니다."}
        st = self.app.run(query="연구용 배터리 5개 수입", submit=True)
        self.assertTrue(any("⏳" in t and "사용량 한도" in t for t in self.app.texts(st.warning)))
        self.assertFalse(any("분석 실패" in t for t in self.app.texts(st.error)))

        self.app.run_result = {"status": "error", "stage": "hsk", "message": "지원 대상 품목을 특정하지 못했습니다."}
        st = self.app.run(query="와인 3개 수입", submit=True)
        self.assertTrue(any("분석 실패 (HSK 추론 단계)" in t for t in self.app.texts(st.error)))

    def test_cache_hit_and_markdown_download_without_server_file(self):
        self.app.run_result = {"status": "success", "report_markdown": "# 보고서", "cache_hit": True}
        st = self.app.run(query="연구용 배터리 5개 수입", submit=True)
        self.assertTrue(any("이전 분석 결과" in t for t in self.app.texts(st.caption)))
        buttons = [c for col in self.app.columns_created for c in col.download_button.call_args_list]
        self.assertEqual(len(buttons), 1)                    # md 만 (서버에 docx 파일이 없음)
        self.assertEqual(buttons[0].kwargs["data"], "# 보고서".encode("utf-8"))
        self.assertEqual(buttons[0].kwargs["mime"], "text/markdown")

    def test_diagnostics_only_with_debug_param(self):
        st = self.app.run()
        st.sidebar.expander.assert_not_called()
        st = self.app.run(query_params={"debug": "1"})
        st.sidebar.expander.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
