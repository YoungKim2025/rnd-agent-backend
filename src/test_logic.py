"""오프라인 단위 테스트 (신규) — API 키·네트워크·벡터 DB 없이 실행된다.

    python -m unittest src/test_logic.py -v        (프로젝트 루트에서)
    python src/test_logic.py

검증 대상: 수량·단가 추출(+입력 해석 보강), 수입 목적 판정, 세액 시나리오, 수량 한도 검증,
           재시도·오류 안내·클라이언트 캐시, 법령 근거 분리, 규정 원문 분할/메타데이터(scope),
           벡터 스토어 구조 버전·캐시·단일 메타데이터 조회, 정합성 점검기, 보고서 end-to-end(LLM·RAG는 가짜로 대체)
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
        self.assertIsNone(rl.extract_quantity("12V 5A 어댑터"))

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
            "대비판매 증가를 위한 판매용 배터리 50개": "commercial",
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
            "상업적 판매 안 해, 연구용이야": "research",
            "연구개발 목적이 아니라 판매용이야": "commercial",
            "재판매 목적 전혀 없고 R&D용": "research",
            "오늘 점심 뭐 먹을까": "unknown",
            "안전 인증 문의": "unknown",
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
        self.assertEqual(rl.check_quantity(25, 1500, "전파법", qc_purpose=True)["state"], "exceeded")
        self.assertEqual(rl.check_quantity(25, 1500, "복합", qc_purpose=True)["state"], "exceeded")
        self.assertEqual(rl.check_quantity(25, 1500, "전안법", qc_purpose=True)["state"], "within")


class RetryAndErrorTests(unittest.TestCase):
    def test_transient_vs_quota(self):
        class ServerError(Exception): pass
        class CodedError(Exception): code = 503

        self.assertTrue(config.is_transient(ServerError("500 INTERNAL")))
        self.assertTrue(config.is_transient(CodedError("x")))
        self.assertTrue(config.is_transient(RuntimeError("503 UNAVAILABLE")))
        self.assertFalse(config.is_transient(RuntimeError("404 NOT_FOUND model")))
        self.assertFalse(config.is_transient(RuntimeError("429 RESOURCE_EXHAUSTED")))
        self.assertTrue(config.is_quota_error(RuntimeError("429 RESOURCE_EXHAUSTED")))

    def test_with_retry_retries_then_succeeds(self):
        calls = {"n": 0}
        def flaky():
            calls["n"] += 1
            if calls["n"] < 3: raise RuntimeError("503 UNAVAILABLE")
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
        self.assertIn("GEMINI_API_KEY", key_missing["message"])

        daily = config.format_api_error(RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 18h32m33.26s"))
        self.assertEqual(daily["error_kind"], "quota")
        self.assertIn("내일", daily["message"])
        
        self.assertEqual(config.format_api_error(RuntimeError("503 UNAVAILABLE"))["error_kind"], "overloaded")
        
        other = config.format_api_error(ValueError("secret detail"), "의도 분류")
        self.assertEqual(other["error_kind"], "other")
        self.assertNotIn("secret detail", other["message"])
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

    def test_normalize_prefix_numbered_headers_and_adjacent_laws(self):
        self.assertEqual(tc.normalize_law_basis("관련 법령: 「전기용품」 제15조(신고 등), 제16조(신고의 면제)"),
                         "「전기용품」 제15조(신고 등), 제16조(신고의 면제)")
        self.assertEqual(tc.normalize_law_basis("「전파법」 제58조의2(적합성평가)「전파법」 제58조의3(면제)"),
                         "「전파법」 제58조의2(적합성평가), 「전파법」 제58조의3(면제)")
        joined = tc.normalize_law_basis("「전파법」 제58조의3(면제) 관련 규정, 관련 법령 2: 「전기용품」 제6조(면제)")
        self.assertIn("관련 규정 및 「전기용품」", joined)

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
        self.assertEqual(items["8507600000"]["law_title"], "제목 - 구라벨")
        self.assertEqual(items["8504400000"]["law_basis"], "「전안법」 제15조(신고)")
        self.assertEqual(items["8504400000"]["item_label"], "전원공급장치")