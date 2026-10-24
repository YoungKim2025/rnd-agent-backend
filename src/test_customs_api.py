"""customs_api.py 단위 테스트 (test_customs_api.py 대체본)

예전 파일은 API 구현 자체가 들어 있고 assert 가 없는 스크립트였다. 구현은 customs_api.py 로 옮겼고,
여기서는 requests.get 을 가짜로 바꿔 네트워크 없이 동작을 검증한다.

    python src/test_customs_api.py
실제 API 확인(선택, 인증키 필요):
    python src/test_customs_api.py --live 8507600000
"""
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests  # noqa: E402

import customs_api  # noqa: E402

TARIFF_XML = """<?xml version="1.0" encoding="UTF-8"?>
<trrtQryRtnVo><tCnt>4</tCnt>
 <TrrtQryRsltVo><trrtTpNm>기본세율</trrtTpNm><trrt>8</trrt></TrrtQryRsltVo>
 <TrrtQryRsltVo><trrtTpNm>WTO협정세율</trrtTpNm><trrt>0</trrt></TrrtQryRsltVo>
 <TrrtQryRsltVo><trrtTpNm>한-미 FTA협정세율</trrtTpNm><trrt>0</trrt></TrrtQryRsltVo>
 <TrrtQryRsltVo><trrtTpNm>한-중 FTA협정세율</trrtTpNm><trrt>1.6</trrt></TrrtQryRsltVo>
</trrtQryRtnVo>""".encode("utf-8")


def fake_response(content: bytes = b"", status: int = 200):
    response = mock.Mock()
    response.status_code = status
    response.content = content
    return response


class CustomsApiTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ, {"CUSTOMS_TARIFF_KEY": "SECRET-KEY-123"}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_missing_key_is_error_not_fake_success(self):
        with mock.patch.dict(os.environ, {"CUSTOMS_TARIFF_KEY": ""}):
            result = customs_api.fetch_tariff_from_customs_api("8507600000")
        self.assertEqual(result["status"], "error")
        self.assertIn("CUSTOMS_TARIFF_KEY", result["message"])

    def test_parses_real_rates_instead_of_hardcoded_8_percent(self):
        xml = TARIFF_XML.replace(b"<trrt>8</trrt>", b"<trrt>13</trrt>")
        with mock.patch("customs_api.requests.get", return_value=fake_response(xml)):
            result = customs_api.fetch_tariff_from_customs_api("8507.60-0000")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["base_rate_val"], 13.0)
        self.assertEqual(result["wto_rate_val"], 0.0)
        self.assertEqual(result["fta_rate_us_val"], 0.0)
        self.assertEqual(result["fta_rate_cn_val"], 1.6)
        self.assertEqual(result["hsk_code"], "8507600000")
        self.assertEqual(len(result["rates"]), 4)

    def test_ssl_verification_is_on_by_default(self):
        with mock.patch("customs_api.requests.get", return_value=fake_response(TARIFF_XML)) as get:
            customs_api.fetch_tariff_from_customs_api("8507600000")
        self.assertIs(get.call_args.kwargs["verify"], True)

    def test_ssl_options_via_env(self):
        with mock.patch.dict(os.environ, {"CUSTOMS_CA_BUNDLE": "C:/certs/ca.pem"}):
            self.assertEqual(customs_api._verify_option(), "C:/certs/ca.pem")
        with mock.patch.dict(os.environ, {"CUSTOMS_CA_BUNDLE": "", "CUSTOMS_VERIFY_SSL": "0"}):
            self.assertFalse(customs_api._verify_option())

    def test_no_base_rate_is_error(self):
        xml = "<r><tCnt>1</tCnt><TrrtQryRsltVo><trrtTpNm>WTO협정세율</trrtTpNm><trrt>0</trrt></TrrtQryRsltVo></r>".encode("utf-8")
        with mock.patch("customs_api.requests.get", return_value=fake_response(xml)):
            result = customs_api.fetch_tariff_from_customs_api("8507600000")
        self.assertEqual(result["status"], "error")

    def test_empty_result_is_not_found(self):
        with mock.patch("customs_api.requests.get", return_value=fake_response(b"<r><tCnt>0</tCnt></r>")):
            self.assertEqual(customs_api.fetch_tariff_from_customs_api("8507600000")["status"], "not_found")

    def test_http_error_and_bad_xml(self):
        with mock.patch("customs_api.requests.get", return_value=fake_response(b"", 500)):
            self.assertEqual(customs_api.fetch_tariff_from_customs_api("8507600000")["status"], "error")
        with mock.patch("customs_api.requests.get", return_value=fake_response(b"not xml")):
            self.assertEqual(customs_api.fetch_tariff_from_customs_api("8507600000")["status"], "error")

    def test_exception_message_does_not_leak_key(self):
        boom = requests.ConnectionError("GET https://x/?crkyCn=SECRET-KEY-123&hsSgn=1 failed")
        with mock.patch("customs_api.requests.get", side_effect=boom):
            result = customs_api.fetch_tariff_from_customs_api("8507600000")
        self.assertEqual(result["status"], "error")
        self.assertNotIn("SECRET-KEY-123", result["message"])

    def test_empty_code_is_error(self):
        self.assertEqual(customs_api.fetch_tariff_from_customs_api("")["status"], "error")


if __name__ == "__main__":
    if "--live" in sys.argv:
        code = sys.argv[sys.argv.index("--live") + 1] if len(sys.argv) > sys.argv.index("--live") + 1 else "8507600000"
        print(customs_api.fetch_tariff_from_customs_api(code))
    else:
        unittest.main(verbosity=2)
