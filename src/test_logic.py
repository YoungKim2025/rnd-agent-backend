"""오프라인 단위 테스트 (신규) — API 키·네트워크·벡터 DB 없이 실행된다.

    python -m unittest src/test_logic.py -v        (프로젝트 루트에서)
    python src/test_logic.py

검증 대상: 수량·단가 추출(+입력 해석 보강), 수입 목적 판정, 세액 시나리오, 수량 한도 검증,
           재시도·오류 안내·클라이언트 캐시, 법령 근거 분리, 규정 원문 분할/메타데이터(scope),
           벡터 스토어 구조 버전·캐시·단일 임베딩, 정합성 점검기, 보고서 end-to-end(LLM·RAG는 가짜로 대체)
"""
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_vector_store as bvs  # noqa: E402
import check_data_consistency as cdc  # noqa: E402
import main_pipeline as mp  # noqa: E402
import rag_store  # noqa: E402
import report_logic as rl  # noqa: E402
import tariff_catalog as tc  # noqa: E402
import config  # noqa: E402

DEFAULT_QUERY = ("연구개발 목적으로 리튬이온 배터리 프로토타입 100개를 수입하려고 해. "
                 "개당 가격은 5천원이고 인증은 아직 없어. 전안법 면제 조건과 관세율 알려줘.")


class QuantityAndPriceTests(unittest.TestCase):
    def test_quantity_basic(self):
        self.assertEqual(rl.extract_quantity(DEFAULT_QUERY), 100)

    def test_quantity_ignores_per_unit_and_months(self):
        self.assertEqual(rl.extract_quantity("1개당 5000원, 20개 필요"), 20)
        self.assertEqual(rl.extract_quantity("2개월 뒤 30대 수입"), 30)

    def test_quantity_missing_is_none(self):
        self.assertIsNone(rl.extract_quantity("연구용 배터리 수입 관세 알려줘"))
        self.assertIsNone(rl.extract_quantity("12V 5A 어댑터"))  # 전압·전류 숫자를 수량으로 오인하지 않음

    def test_english_unit_not_matched_inside_word(self):
        self.assertEqual(rl.find_quantities("$10 each 5개"), [5])

    def test_price_unit(self):
        self.assertEqual(rl.extract_unit_price(DEFAULT_QUERY, 100), {"status": "found", "unit_krw": 5000, "basis": "unit"})

    def test_price_korean_numbers(self):
        cases = {"대당 12,000원 7대": 12000, "개당 1만5천원": 15000, "개당 2천만원": 20_000_000,
                 "개당 만원": 10_000, "개당 1억 2천만원": 120_000_000}
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(rl.extract_unit_price(text, 1)["unit_krw"], expected)

    def test_price_total_divided_by_quantity(self):
        self.assertEqual(rl.extract_unit_price("총 500만원에 25개", 25), {"status": "found", "unit_krw": 200_000, "basis": "total"})

    def test_price_unspecified_is_flagged(self):
        self.assertEqual(rl.extract_unit_price("배터리 10대 1만5천원", 10)["basis"], "unspecified")

    def test_price_foreign_and_missing(self):
        self.assertEqual(rl.extract_unit_price("개당 $10", 5)["status"], "foreign")
        self.assertEqual(rl.extract_unit_price("개당 12달러", 5)["status"], "foreign")
        self.assertEqual(rl.extract_unit_price("배터리 3대 수입", 3)["status"], "missing")
        self.assertEqual(rl.extract_unit_price("원본 서류 5대", 5)["status"], "missing")


class InputInterpretationTests(unittest.TestCase):
    def test_non_prefix_is_not_commercial(self):
        cases = {
            "비상업적 연구 목적으로 배터리 5개": "research",
            "비판매용 연구 배터리 5개": "research",
            "판매 목적 아님, 연구용 배터리 5개": "research",
            "대비판매 증가를 위한 판매용 배터리 50개": "commercial",   # '대비' 는 부정 접두가 아니다
            "판매할 목적으로 배터리 50개": "commercial",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(rl.detect_purpose(text), expected)

    def test_country_year_place_counters_are_not_quantities(self):
        self.assertEqual(rl.find_quantities("3개국에서 배터리 10개"), [10])
        self.assertEqual(rl.find_quantities("3개년 계획으로 20대"), [20])
        self.assertEqual(rl.find_quantities("2개소 연구실에 8개"), [8])

    def test_budget_or_cap_is_not_a_unit_price(self):
        for text in ("예산 500만원 이내로 배터리 25개", "개당 5천원 이하로 알고 있어", "한도 100만원까지"):
            with self.subTest(text=text):
                self.assertEqual(rl.extract_unit_price(text, 25)["status"], "missing")
        mixed = rl.extract_unit_price("개당 5천원, 총 예산 100만원 이내", 25)
        self.assertEqual((mixed["status"], mixed["unit_krw"]), ("found", 5000))


class PurposeTests(unittest.TestCase):
    def test_cases(self):
        cases = {
            "판매용이 아닌 연구개발용으로 쓸 태블릿 PC 5대": "research",
            "판매할 목적으로 태블릿 PC 50대": "commercial",
            "상업적 판매 안 해, 연구용이야": "research",       # '안 해' 띄어쓴 부정
            "연구개발 목적이 아니라 판매용이야": "commercial",
            "재판매 목적 전혀 없고 R&D용": "research",
            "오늘 점심 뭐 먹을까": "unknown",
            "안전 인증 문의": "unknown",                       # '안전'의 '안'은 부정이 아님
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(rl.detect_purpose(text), expected)

    def test_origin_cert(self):
        self.assertTrue(rl.mentions_origin_cert("원산지증명서 있어"))
        self.assertFalse(rl.mentions_origin_cert("원산지증명서 없어"))
        self.assertFalse(rl.mentions_origin_cert(DEFAULT_QUERY))


class TaxAndQuantityTests(unittest.TestCase):
    def test_scenarios_default_to_base_rate(self):
        tax = rl.build_tax_scenarios(8, 0, 25, 1_000_000)
        self.assertEqual(tax["applied_key"], "base")
        self.assertEqual((tax["base"]["estimated_duty"], tax["base"]["estimated_vat"], tax["base"]["total_tax"]),
                         (2_000_000, 2_700_000, 4_700_000))
        self.assertEqual(tax["fta"]["total_tax"], 2_500_000)

    def test_fta_applied_only_when_requested(self):
        self.assertEqual(rl.build_tax_scenarios(8, 0, 25, 1_000_000, fta_applied=True)["applied_key"], "fta")

    def test_quantity_check_states(self):
        self.assertEqual(rl.check_quantity(25, 0, "전안법")["state"], "unregistered")
        self.assertEqual(rl.check_quantity(25, 1500, "전파법")["state"], "within")
        self.assertEqual(rl.check_quantity(1501, 1500, "전파법")["state"], "exceeded")
        self.assertEqual(rl.check_quantity(25, 1500, "전파법", qc_purpose=True)["state"], "exceeded")  # 품질검사 10대 기준
        self.assertEqual(rl.check_quantity(25, 1500, "복합", qc_purpose=True)["state"], "exceeded")    # 복합 품목도 전파법 10대 기준 적용
        self.assertEqual(rl.check_quantity(25, 1500, "전안법", qc_purpose=True)["state"], "within")    # 10대 기준은 전파법 계열만


class RetryAndErrorTests(unittest.TestCase):
    def test_transient_vs_quota(self):
        class ServerError(Exception):
            pass

        class CodedError(Exception):
            code = 503

        self.assertTrue(config.is_transient(ServerError("500 INTERNAL")))
        self.assertTrue(config.is_transient(CodedError("x")))
        self.assertTrue(config.is_transient(RuntimeError("503 UNAVAILABLE")))
        self.assertFalse(config.is_transient(RuntimeError("404 NOT_FOUND model")))
        # 429 는 재시도해도 소용없으므로 transient 가 아니라 quota
        self.assertFalse(config.is_transient(RuntimeError("429 RESOURCE_EXHAUSTED")))
        self.assertTrue(config.is_quota_error(RuntimeError("429 RESOURCE_EXHAUSTED")))

    def test_with_retry_retries_then_succeeds(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("503 UNAVAILABLE")
            return "ok"

        with mock.patch.object(config.time, "sleep"):
            self.assertEqual(config.with_retry(flaky), "ok")
        self.assertEqual(calls["n"], 3)

    def test_with_retry_gives_up_after_max_retries(self):
        calls = {"n": 0}

        def always_down():
            calls["n"] += 1
            raise RuntimeError("503 UNAVAILABLE")

        with mock.patch.object(config.time, "sleep"), self.assertRaises(RuntimeError):
            config.with_retry(always_down)
        self.assertEqual(calls["n"], config.MAX_RETRIES)

    def test_with_retry_does_not_retry_quota(self):
        calls = {"n": 0}

        def quota():
            calls["n"] += 1
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

        with mock.patch.object(config.time, "sleep"), self.assertRaises(RuntimeError):
            config.with_retry(quota)
        self.assertEqual(calls["n"], 1)

    def test_format_api_error_kinds(self):
        key_missing = config.format_api_error(RuntimeError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다."), "HSK 추론")
        self.assertEqual(key_missing["error_kind"], "config")
        self.assertIn("GEMINI_API_KEY", key_missing["message"])        # 키 누락 원인이 가려지지 않는다

        daily = config.format_api_error(RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 18h32m33.26s"))
        self.assertEqual(daily["error_kind"], "quota")
        self.assertIn("내일", daily["message"])
        short = config.format_api_error(RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 30.5s"))
        self.assertIn("약 30초", short["message"])

        self.assertEqual(config.format_api_error(RuntimeError("503 UNAVAILABLE"))["error_kind"], "overloaded")
        other = config.format_api_error(ValueError("secret detail"), "의도 분류")
        self.assertEqual(other["error_kind"], "other")
        self.assertNotIn("secret detail", other["message"])           # 원문은 숨긴다
        self.assertIn("의도 분류", other["message"])

    def test_get_client_is_cached(self):
        sentinel = object()
        with mock.patch.object(config, "_client", None), mock.patch.object(config, "make_client", return_value=sentinel) as made:
            self.assertIs(config.get_client(), sentinel)
            self.assertIs(config.get_client(), sentinel)
        self.assertEqual(made.call_count, 1)


class CatalogLawBasisTests(unittest.TestCase):
    RAW_BATTERY = "「전기용품 및 생활용품 안전관리법」 제6조(안전인증의 면제) 및 제16조(안전확인신고의 면제)및 관련 하위 법령 - 배터리류"

    def test_split_and_normalize(self):
        basis, label = tc.split_law_title(self.RAW_BATTERY)
        self.assertEqual(label, "배터리류")
        self.assertEqual(tc.normalize_law_basis(basis),
                         "「전기용품 및 생활용품 안전관리법」 제6조(안전인증의 면제) 및 제16조(안전확인신고의 면제) 및 관련 하위 법령")
        self.assertEqual(tc.split_law_title("라벨 없는 제목"), ("라벨 없는 제목", ""))

    def test_normalize_prefix_numbered_headers_and_adjacent_laws(self):
        self.assertEqual(tc.normalize_law_basis("관련 법령: 「전기용품」 제15조(신고 등), 제16조(신고의 면제)"),
                         "「전기용품」 제15조(신고 등), 제16조(신고의 면제)")
        self.assertEqual(tc.normalize_law_basis("「전파법」 제58조의2(적합성평가)「전파법」 제58조의3(면제)"),
                         "「전파법」 제58조의2(적합성평가), 「전파법」 제58조의3(면제)")
        joined = tc.normalize_law_basis("「전파법」 제58조의3(면제) 관련 규정, 관련 법령 2: 「전기용품」 제6조(면제)")
        self.assertIn("관련 규정 및 「전기용품」", joined)
        self.assertNotIn("관련 법령 2", joined)

    def test_article_numbers_are_never_changed(self):
        before = set(re.findall(r"제\d+조(?:의\d+)?", self.RAW_BATTERY))
        after = set(re.findall(r"제\d+조(?:의\d+)?", tc.normalize_law_basis(tc.split_law_title(self.RAW_BATTERY)[0])))
        self.assertEqual(before, after)

    def test_optional_columns_override_derived_values(self):
        header = "hsk_code,item_name,law_type,law_title,limit_quantity,basic_rate,fta_rate,law_basis,item_label\n"
        rows = ('8507600000,배터리,전안법,"제목 - 구라벨",0,8,0,"「전안법」 제6조(안전인증의 면제)",새라벨\n'
                '8504400000,어댑터,전안법,"「전안법」 제15조(신고) - 전원공급장치",0,8,0,,\n')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.csv"
            path.write_text(header + rows, encoding="utf-8")
            with mock.patch.object(tc, "CSV_PATH", path), mock.patch.dict(tc._cache, {"key": None, "items": None}):
                items = tc.get_catalog()
        self.assertEqual(items["8507600000"]["law_basis"], "「전안법」 제6조(안전인증의 면제)")
        self.assertEqual(items["8507600000"]["item_label"], "새라벨")
        self.assertEqual(items["8507600000"]["law_title"], "제목 - 구라벨")        # 원문은 보존
        self.assertEqual(items["8504400000"]["law_basis"], "「전안법」 제15조(신고)")   # 열이 비면 law_title 에서 유도
        self.assertEqual(items["8504400000"]["item_label"], "전원공급장치")

    def test_repository_catalog_has_clean_law_basis(self):
        for entry in tc.get_catalog().values():
            self.assertTrue(entry["law_basis"], entry["hsk_code"])
            self.assertNotIn(" - ", entry["law_basis"], entry["hsk_code"])
            self.assertNotRegex(entry["law_basis"], r"\)(?=[및와과])", entry["hsk_code"])


class RagStoreTests(unittest.TestCase):
    class Doc:
        def __init__(self, section, scope):
            self.page_content = f"조항 {section}"
            self.metadata = {"section": section, "part": 0, "scope": scope, "title": f"{section}. 제목", "source": "kc_exemption.txt"}

    class FakeStore:
        def __init__(self):
            self.vector_calls, self.text_calls = [], []

        def _result(self, flt):
            text = str(flt)
            return [RagStoreTests.Doc(2, "specific")] if "hsk6_" in text else [RagStoreTests.Doc(7, "general"), RagStoreTests.Doc(8, "general")]

        def similarity_search_by_vector(self, vector, k=4, filter=None):
            self.vector_calls.append((vector, k, filter))
            return self._result(filter)

        def similarity_search(self, query, k=4, filter=None):
            self.text_calls.append((query, k, filter))
            return self._result(filter)

    class FakeEmbedder:
        def __init__(self):
            self.calls = 0

        def embed_query(self, text):
            self.calls += 1
            return [0.1, 0.2]

    def test_one_embedding_for_both_searches_and_scope_filter(self):
        store, embedder = self.FakeStore(), self.FakeEmbedder()
        docs = rag_store.retrieve_legal_docs("배터리 면제", "8507600000", "전안법", store=store, embedder=embedder)
        self.assertEqual(embedder.calls, 1)                 # 예전에는 질문 임베딩이 2회 호출됨
        self.assertEqual(len(store.vector_calls), 2)
        self.assertEqual([d.metadata["section"] for d in docs], [2, 7, 8])
        general_filter = store.vector_calls[1][2]["$and"]
        self.assertIn({"scope": "general"}, general_filter)  # 다른 품목의 전용 조항이 섞이지 않는다
        law_types = next(c["law_type"]["$in"] for c in general_filter if "law_type" in c)
        self.assertNotIn("radio", law_types)                # 전안법 품목에는 전파법 일반 조항을 섞지 않는다
        self.assertIn("common", law_types)

    def test_text_search_path_when_no_embedder(self):
        store = self.FakeStore()
        rag_store.retrieve_legal_docs("q", "8507600000", "전안법", store=store)
        self.assertEqual(len(store.text_calls), 2)
        self.assertEqual(store.vector_calls, [])

    def test_check_store_meta(self):
        signature = {"backend": "gemini", "model": "gemini-embedding-001"}

        def check(meta):
            with tempfile.TemporaryDirectory() as tmp:
                if meta is not None:
                    (Path(tmp) / rag_store.META_NAME).write_text(json.dumps(meta), encoding="utf-8")
                return rag_store.check_store_meta(signature, chroma_dir=Path(tmp))

        self.assertIsNone(check({**signature, "schema": bvs.SCHEMA_VERSION}))
        with self.assertRaisesRegex(rag_store.VectorStoreError, "서명 파일"):
            check(None)
        with self.assertRaisesRegex(rag_store.VectorStoreError, "구조 버전"):
            check(dict(signature))                                  # schema 없음 (구버전 DB)
        with self.assertRaisesRegex(rag_store.VectorStoreError, "구조 버전"):
            check({**signature, "schema": bvs.SCHEMA_VERSION - 1})
        with self.assertRaisesRegex(rag_store.VectorStoreError, "현재 설정"):
            check({"backend": "hf", "model": "x", "schema": bvs.SCHEMA_VERSION})

    def test_store_is_cached_until_db_changes(self):
        loads = []

        def fake_load():
            loads.append(1)
            return object()

        key = {"v": ("a", "b")}
        with mock.patch.dict(rag_store._cache, {"key": None, "store": None}), \
                mock.patch.object(rag_store, "load_vectorstore", side_effect=fake_load), \
                mock.patch.object(rag_store, "_cache_key", side_effect=lambda: key["v"]):
            first = rag_store.get_store()
            self.assertIs(rag_store.get_store(), first)
            self.assertEqual(len(loads), 1)
            key["v"] = ("a", "c")                                   # DB 재구축으로 파일 시각이 바뀜
            self.assertIsNot(rag_store.get_store(), first)
            self.assertEqual(len(loads), 2)

    def test_schema_version_shared_with_builder(self):
        self.assertIsInstance(bvs.SCHEMA_VERSION, int)
        self.assertIs(rag_store.SCHEMA_VERSION, bvs.SCHEMA_VERSION)


class SectionAndMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = bvs.DATA_PATH.read_text(encoding="utf-8")
        cls.docs = bvs.build_documents(cls.text)

    def test_sections_split_on_numbered_headers_only(self):
        sections = bvs.parse_sections(self.text)
        self.assertEqual([s["number"] for s in sections], list(range(len(sections))))
        procedure = next(s for s in sections if "절차" in s["title"])
        self.assertIn("1. 면제 확인(신청)서", procedure["body"])  # 들여쓴 하위 번호는 조항 머리가 아님

    def test_general_radio_rule_is_not_orphaned(self):
        # 태블릿 1,500대 기준 조항은 HSK 코드가 없지만 law_type=radio 로 검색되어야 한다
        general = [m for c, m in self.docs if "1,500대 이하" in c]
        self.assertTrue(general)
        self.assertEqual(general[0]["law_type"], "radio")
        self.assertIn("radio", rag_store.allowed_law_types("전파법"))

    def test_every_hsk_in_specific_sections_is_flagged(self):
        # 데이터 파일 내용에 의존하지 않도록, 원문에 실제로 나오는 10자리 코드를 기준으로 검증한다
        flags = {k for _, m in self.docs for k in m if k.startswith("hsk6_")}
        for section in bvs.parse_sections(self.text):
            if bvs.classify_law_type(section["number"], section["title"], section["body"]) == "common":
                continue
            for code in bvs.extract_hsk_codes(f"{section['title']}\n{section['body']}"):
                self.assertIn(f"hsk6_{code[:6]}", flags, f"§{section['number']} {code}")

    def test_common_sections_have_no_hsk_flags(self):
        for content, meta in self.docs:
            if meta["law_type"] == "common":
                self.assertFalse([k for k in meta if k.startswith("hsk6_")], meta["title"])

    def test_scope_metadata(self):
        for content, meta in self.docs:
            has_flag = any(k.startswith("hsk6_") for k in meta)
            self.assertEqual(meta["scope"], "specific" if has_flag else "general", meta["title"])
        general_titles = [m["title"] for _, m in self.docs if m["scope"] == "general"]
        self.assertTrue(any("1,500대" in c for c, m in self.docs if m["scope"] == "general"))   # 전파법 일반 기준은 general
        self.assertTrue(any("절차" in t for t in general_titles))

    def test_chunk_size_and_metadata_types(self):
        for content, meta in self.docs:
            self.assertLessEqual(len(content), bvs.MAX_CHARS + 50)
            for value in meta.values():
                self.assertIsInstance(value, (str, int, float, bool))

    def test_long_section_is_split_with_header(self):
        header = "9. 긴 조항"
        body = "\n".join(f"- 항목 {i} " + "가" * 60 for i in range(40))
        parts = bvs._split_body(header, body, 500)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(p.startswith(header) and len(p) <= 560 for p in parts))


class ConsistencyCheckerTests(unittest.TestCase):
    def _entry(self, code, law_type, law_title, limit=0):
        return {"hsk_code": code, "item_name": "테스트", "law_type": law_type, "law_title": law_title, "limit_quantity": limit}

    def test_detects_hsk_and_article_and_limit_mismatch(self):
        catalog = {
            "8504400000": self._entry("8504400000", "전안법", "전기용품 및 생활용품 안전관리법 제6조(안전인증의 면제) - 전원"),
            "8806210000": self._entry("8806210000", "전파법", "전파법 제58조의3(적합성평가의 면제) - 드론", limit=0),
        }
        sections = [
            {"number": 3, "title": "어댑터 (HSK: 8504403000)", "body": "- 근거 법령: 「전기용품 및 생활용품 안전관리법」 제9조(안전인증의 면제)"},
            {"number": 4, "title": "드론 (HSK: 8806210000)", "body": "- 관련 법령: 전파법 제58조의3(적합성평가의 면제)\n- 수량 제한: 100대 이하"},
        ]
        findings = cdc.run_checks(catalog, sections)
        errors = [m for lvl, m in findings if lvl == "ERROR"]
        warns = [m for lvl, m in findings if lvl == "WARN"]
        self.assertTrue(any("8504403000" in m and "8504400000" in m for m in errors))      # HSK 10자리 불일치
        self.assertTrue(any("제6조" in m and "제9조" in m for m in errors))                 # 조문 번호 불일치
        self.assertTrue(any("100" in m and "미등록" in m for m in warns))                  # 한도 미등록 vs 원문 100대
        self.assertFalse(any("58" in m for m in errors))                                    # 일치하는 조문은 오류 아님

    def test_checker_runs_on_repository_data(self):
        # 실제 데이터의 불일치 여부는 `python src/check_data_consistency.py` 로 확인한다.
        # (단위 테스트가 사용자의 데이터 편집 상태 때문에 깨지지 않도록 '실행 가능'만 검증)
        findings = cdc.run_checks()
        self.assertTrue(findings)
        self.assertTrue(all(level in ("ERROR", "WARN", "INFO") for level, _ in findings))


class EndToEndReportTests(unittest.TestCase):
    """LLM·벡터 DB를 가짜로 바꾸고 파일 생성까지 확인한다."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        patches = [
            mock.patch.object(mp, "OUTPUT_DIR", Path(self.tmp.name)),
            mock.patch.object(mp, "retrieve_context", return_value=("규정 발췌", ["2. 리튬이온 축전지 및 배터리 팩 (HSK: 8507600000) (kc_exemption.txt)"])),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)

    def run_report(self, query, hsk="8507600000", llm_text="금번 수입 물품은 연구개발용 시제품이며 상업적 유통을 금지하고 사후 추적 관리를 이행합니다."):
        with mock.patch.object(mp.config, "generate_text", return_value=llm_text):
            return mp.generate_fast_track_report(query, hsk)

    def test_default_query_uses_user_price(self):
        result = self.run_report(DEFAULT_QUERY)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["unit_price_krw"], 5000)
        self.assertEqual(result["tax"]["total_krw"], 500_000)      # 예전에는 100,000,000 으로 계산됨
        self.assertEqual(result["tax"]["applied_key"], "base")
        self.assertTrue(Path(result["md_path"]).exists() and Path(result["docx_path"]).exists())
        self.assertIn("참고용 초안", result["report_markdown"])
        self.assertTrue(any("한도" in w for w in result["warnings"]))   # 전안법 품목: 한도 미등록 경고

    def test_unregistered_limit_message_everywhere(self):
        # 확인이 필요한 사항 / 사유서 말미 모두 같은 문구, LLM 성공·실패 두 경로 모두
        for llm_text in ("연구개발용 시제품이며 상업적 유통을 금지하고 사후 추적 관리를 이행합니다.", None):
            with self.subTest(llm="ok" if llm_text else "fail"):
                if llm_text:
                    result = self.run_report(DEFAULT_QUERY, llm_text=llm_text)
                else:
                    with mock.patch.object(mp.config, "generate_text", side_effect=RuntimeError("503")):
                        result = mp.generate_fast_track_report(DEFAULT_QUERY, "8507600000")
                self.assertIn(rl.UNREGISTERED_LIMIT_MESSAGE, result["warnings"])
                section3 = result["report_markdown"].split("### 3.")[1].split("### 4.")[0]
                self.assertTrue(section3.replace("---", "").strip().endswith(f"※ {rl.UNREGISTERED_LIMIT_MESSAGE}"))

    def test_law_basis_has_no_item_label_in_report(self):
        result = self.run_report(DEFAULT_QUERY)
        md = result["report_markdown"]
        self.assertNotIn("- 배터리류", md)                # 품목 라벨은 사유서 제목·본문에 들어가지 않는다
        self.assertNotIn("면제)및", md)
        title_line = next(line for line in md.splitlines() if "[문서 제목]" in line)
        self.assertIn("제6조(안전인증의 면제)", title_line)
        self.assertTrue(title_line.rstrip("*").endswith("신청 사유서"))

    def test_pledge_is_guaranteed_when_llm_omits_it(self):
        no_pledge = self.run_report(DEFAULT_QUERY, llm_text="본 물품은 연구개발용 시제품으로 면제를 신청합니다.")
        self.assertEqual(no_pledge["report_markdown"].count(mp.PLEDGE_SENTENCE), 1)
        with_pledge = self.run_report(
            DEFAULT_QUERY,
            llm_text="연구개발용 시제품이며 상업적 판매를 하지 않을 것을 서약합니다. 사후 추적 관리도 이행합니다.")
        self.assertEqual(with_pledge["report_markdown"].count(mp.PLEDGE_SENTENCE), 0)   # 이미 있으면 중복해서 붙이지 않는다

    def test_pledge_present_in_fallback_and_before_limit_note(self):
        with mock.patch.object(mp.config, "generate_text", side_effect=RuntimeError("503")):
            result = mp.generate_fast_track_report(DEFAULT_QUERY, "8507600000")
        section3 = result["report_markdown"].split("### 3.")[1].split("### 4.")[0]
        self.assertEqual(section3.count(mp.PLEDGE_SENTENCE), 1)
        self.assertLess(section3.index(mp.PLEDGE_SENTENCE), section3.index("※"))   # 한도 안내(※)는 항상 맨 끝

    def test_missing_price_is_flagged_as_assumption(self):
        result = self.run_report("연구개발 목적으로 리튬이온 배터리 25개 수입")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["unit_price_krw"], rl.DEFAULT_UNIT_PRICE_KRW)
        self.assertTrue(any("가정값" in w for w in result["warnings"]))
        self.assertIn("가정값", result["report_markdown"])

    def test_foreign_price_is_rejected(self):
        result = self.run_report("연구개발 목적 배터리 10개, 개당 $20")
        self.assertEqual(result["status"], "error")
        self.assertIn("원화", result["message"])

    def test_missing_quantity_is_rejected(self):
        self.assertEqual(self.run_report("연구개발 목적으로 배터리 수입하려는데 개당 5천원")["status"], "error")

    def test_commercial_is_blocked(self):
        self.assertEqual(self.run_report("판매할 목적으로 배터리 50개 수입")["status"], "error")

    def test_unknown_purpose_has_no_exemption_text(self):
        result = self.run_report("리튬이온 배터리 10개 수입, 개당 1만원")
        self.assertEqual(result["status"], "success")
        self.assertIn("작성하지 않았습니다", result["report_markdown"])
        self.assertNotIn("신청 사유서", result["report_markdown"])

    def test_quantity_over_limit_blocks_reason(self):
        result = self.run_report("연구개발 목적으로 태블릿 PC 2000대 수입, 개당 30만원", hsk="8471300000")
        self.assertEqual(result["status"], "error")
        self.assertIn("초과", result["message"])

    def test_within_limit_states_verified_sentence(self):
        result = self.run_report("연구개발 목적으로 태블릿 PC 30대 수입, 개당 30만원", hsk="8471300000",
                                 llm_text="요청 수량은 법정 기준을 충족하며 연구개발용 시제품으로 상업적 판매는 하지 않습니다.")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["tax"]["base"]["estimated_duty"], 0)  # 태블릿 기본세율 0%

    def test_origin_cert_switches_to_fta(self):
        result = self.run_report("연구개발 목적으로 리튬이온 배터리 25개 수입, 개당 1만원, 원산지증명서 있어")
        self.assertEqual(result["tax"]["applied_key"], "fta")
        self.assertTrue(any("FTA" in w for w in result["warnings"]))

    def test_overclaiming_llm_text_is_replaced(self):
        # 한도 미등록(전안법)인데 LLM 이 '수량 기준 충족'이라고 쓰면 표준 문구로 교체
        result = self.run_report(DEFAULT_QUERY, llm_text="요청 수량은 법정 기준을 충족하며 연구개발용으로 사용합니다.")
        section3 = result["report_markdown"].split("### 3.")[1].split("### 4.")[0]
        self.assertNotIn("기준을 충족", section3)               # LLM 의 '기준 충족' 단정은 제거
        self.assertIn(rl.UNREGISTERED_LIMIT_MESSAGE, section3)  # 대신 한도 미규정 안내가 붙는다

    def test_invented_article_number_is_replaced(self):
        result = self.run_report(DEFAULT_QUERY, llm_text="본 물품은 동법 제99조에 따라 면제됩니다. 연구개발 시제품입니다.")
        self.assertNotIn("제99조", result["report_markdown"])

    def test_llm_failure_falls_back(self):
        with mock.patch.object(mp.config, "generate_text", side_effect=RuntimeError("503")):
            result = mp.generate_fast_track_report(DEFAULT_QUERY, "8507600000")
        self.assertEqual(result["status"], "success")
        self.assertIn("서약", result["report_markdown"])

    def test_rag_failure_adds_warning_but_succeeds(self):
        with mock.patch.object(mp, "retrieve_context", side_effect=RuntimeError("no store")):
            result = self.run_report(DEFAULT_QUERY)
        self.assertEqual(result["status"], "success")
        self.assertTrue(any("규정 검색에 실패" in w for w in result["warnings"]))

    def test_output_filenames_are_unique(self):
        first = self.run_report(DEFAULT_QUERY)
        second = self.run_report(DEFAULT_QUERY)
        self.assertNotEqual(first["md_path"], second["md_path"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
