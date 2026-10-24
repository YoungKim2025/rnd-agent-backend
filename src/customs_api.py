"""관세청 UNI-PASS Open API 클라이언트 (customs_api.py 대체본)

변경점
 - 예전 customs_api.py 는 응답을 읽지 않고 항상 8%/0%를 'success' 로 돌려주는 껍데기였다.
   실제 구현은 test_customs_api.py 에만 있었다 → 이 파일로 옮기고, 가짜 기본값(8.0, '관세청 API 실시간 연동 품목')은 제거.
 - 실패는 항상 {"status": "error", "message": ...} 로 돌려준다 (None 반환 없음). 기본세율을 못 찾으면 error.
 - SSL 검증 기본 ON. (verify=False 제거) 인증서 문제가 있으면 CUSTOMS_CA_BUNDLE=<인증서 경로> 를 설정한다.
   부득이하면 CUSTOMS_VERIFY_SSL=0 (권장하지 않음: 인증키가 쿼리스트링에 실린다)
 - .env 직접 파싱 제거: config.py 가 python-dotenv 로 읽는다.
 - 예외 메시지에서 인증키를 가린다.

환경 변수: CUSTOMS_TARIFF_KEY(관세율 기본 조회), CUSTOMS_HS_SEARCH_KEY(HS부호 검색), CUSTOMS_HS_NAV_KEY(HS 내비게이션)
"""
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402,F401  (import 시 .env 로드)

BASE_URL = "https://unipass.customs.go.kr:38010/ext/rest"
TIMEOUT_S = 10

_US_RE = re.compile(r"미국|한[-\s]?미|\bus\b")
_CN_RE = re.compile(r"중국|한[-\s]?중|\bcn\b")


def _verify_option():
    bundle = os.getenv("CUSTOMS_CA_BUNDLE", "").strip()
    if bundle:
        return bundle
    return os.getenv("CUSTOMS_VERIFY_SSL", "1").strip().lower() not in ("0", "false", "no")


def _redact(message: str, key: str) -> str:
    return message.replace(key, "***") if key else message


def _clean_code(code: str) -> str:
    return re.sub(r"\D", "", str(code or ""))


def _request_xml(endpoint: str, key_env: str, params: dict):
    """(xml root, None) 또는 (None, 오류 dict)."""
    api_key = os.getenv(key_env, "").strip()
    if not api_key:
        return None, {"status": "error", "message": f"API 인증키가 설정되지 않았습니다 ({key_env})."}
    try:
        response = requests.get(
            f"{BASE_URL}/{endpoint}",
            params={"crkyCn": api_key, **params},
            timeout=TIMEOUT_S,
            verify=_verify_option(),
        )
    except requests.RequestException as exc:
        return None, {"status": "error", "message": f"관세청 API 호출 실패: {_redact(str(exc), api_key)[:200]}"}
    if response.status_code != 200:
        return None, {"status": "error", "message": f"HTTP 오류 코드: {response.status_code}"}
    try:
        return ET.fromstring(response.content), None
    except ET.ParseError:
        return None, {"status": "error", "message": "관세청 API 응답(XML)을 해석하지 못했습니다."}


def _to_rate(text: str):
    cleaned = str(text or "").replace("%", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _count(root) -> int:
    try:
        return int((root.findtext("tCnt") or "0").strip())
    except ValueError:
        return 0


def fetch_tariff_from_customs_api(hsk_code: str) -> dict:
    """관세율 기본 조회 (API030). 기본세율을 확인할 수 없으면 status=error."""
    code = _clean_code(hsk_code)
    if not code:
        return {"status": "error", "message": "HSK 코드가 비어 있습니다."}
    root, error = _request_xml("trrtQry/retrieveTrrt", "CUSTOMS_TARIFF_KEY", {"hsSgn": code})
    if error:
        return error
    if _count(root) <= 0:
        return {"status": "not_found", "message": f"관세청 API에서 조회된 관세율 정보가 없습니다: {code}"}

    rates = []
    base = wto = fta_us = fta_cn = None
    for vo in root.findall(".//TrrtQryRsltVo"):
        type_name = (vo.findtext("trrtTpNm") or "").strip()
        rate = _to_rate(vo.findtext("trrt"))
        if rate is None:
            continue
        rates.append({"type": type_name, "rate": rate})
        lowered = type_name.lower()
        if "기본" in lowered and base is None:
            base = rate
        elif "wto" in lowered and wto is None:
            wto = rate
        elif _US_RE.search(lowered) and fta_us is None:
            fta_us = rate
        elif _CN_RE.search(lowered) and fta_cn is None:
            fta_cn = rate

    if base is None:
        return {"status": "error", "message": "응답에서 기본세율 항목을 찾지 못했습니다.", "hsk_code": code, "rates": rates}
    return {
        "status": "success",
        "source": "customs_unipass_api",
        "hsk_code": code,
        "base_rate_val": base,
        "base_rate": f"{base:g}%",
        "wto_rate_val": wto,
        "fta_rate_us_val": fta_us,
        "fta_rate_cn_val": fta_cn,
        "rates": rates,  # 응답에 있던 모든 세율 구분 (원본 유지)
    }


def search_hs_code(hs_sgn: str = None, prnm: str = None, koen_tp: int = 1) -> dict:
    """HS부호 검색 (API018). 부호 또는 물품명으로 검색."""
    params = {"koenTp": koen_tp}
    if hs_sgn:
        params["hsSgn"] = _clean_code(hs_sgn)
    if prnm:
        params["prnm"] = prnm
    root, error = _request_xml("hsSgnQry/searchHsSgn", "CUSTOMS_HS_SEARCH_KEY", params)
    if error:
        return error
    if _count(root) <= 0:
        return {"status": "not_found", "message": "조회된 HS부호 정보가 없습니다."}
    results = [
        {
            "hs_sgn": vo.findtext("hsSgn", ""),
            "kore_prnm": vo.findtext("korePrnm", ""),
            "engl_prnm": vo.findtext("englPrnm", ""),
            "txrt": vo.findtext("txrt", ""),
            "wght_ut": vo.findtext("wghtUt", ""),
            "txtp_sgn": vo.findtext("txtpSgn", ""),
        }
        for vo in root.findall(".//hsSgnSrchRsltVo")
    ]
    return {"status": "success", "source": "customs_api_hs_search", "results": results}


def get_hs_navigation(hs_code: str) -> dict:
    """HS CODE 내비게이션 (API043). 10단위 부호 신고 건수 순위."""
    root, error = _request_xml("cmtrStatsQry/retrieveCmtrStats", "CUSTOMS_HS_NAV_KEY", {"hsSgn": _clean_code(hs_code)})
    if error:
        return error
    if _count(root) <= 0:
        return {"status": "not_found", "message": "조회된 내비게이션(통계) 정보가 없습니다."}
    results = [
        {
            "hs10_sgn": vo.findtext("hs10Sgn", ""),
            "prlst_nm": vo.findtext("prlstNm", ""),
            "acrs_tcnt_rnk": vo.findtext("acrsTcntRnk", ""),
            "prlst_ln_cnt": vo.findtext("prlstLnCnt", ""),
        }
        for vo in root.findall(".//cmtrStatsQryRsltVo")
    ]
    return {"status": "success", "source": "customs_api_navigation", "results": results}
