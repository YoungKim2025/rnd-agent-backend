"""보고서 계산·판정 로직 (신규 파일, 외부 패키지 의존 없음)

main_pipeline.py 에서 분리한 순수 함수 모음이라 API 키·네트워크 없이 단위 테스트할 수 있다.

 1) 수량·단가 추출      : extract_quantity / find_quantities / extract_unit_price
 2) 수입 목적 판정      : detect_purpose (부정 표현 '안 해' 처리 추가), is_qc_purpose
 3) 세액 계산           : calculate_estimated_tax / build_tax_scenarios (기본세율·FTA 두 시나리오)
 4) 수량 한도 검증      : check_quantity (품목표 limit_quantity 사용)
 5) 원산지증명서 언급   : mentions_origin_cert

입력 해석 보강
   - '비상업적', '비판매용' 은 상업 목적으로 오판하지 않는다 (비 + 상업/판매/유통/양산/납품 → '… 아님')
   - '판매 목적 아님' 도 부정으로 인식한다
   - '3개국', '3개년', '3개소' 는 수량으로 보지 않는다
   - '예산 500만원 이내', '개당 5천원 이하' 처럼 상한·예산을 말한 금액은 단가로 쓰지 않는다 (단가 미입력으로 처리 → 가정값 경고)
"""
import re

# ── 수입 목적 판별 ──────────────────────────────────────────────────────────
RESEARCH_KEYWORDS = ("연구", "개발", "r&d", "시제품", "프로토타입", "시험", "테스트")
COMMERCIAL_KEYWORDS = ("판매", "상업", "양산", "납품", "재판매", "유통")
NEGATION_MARKERS = ("아닌", "아니", "아닙", "아님", "않", "없")
NEGATION_WINDOW = 12
NEUTRAL_PHRASES = ("문제없", "상관없", "관계없", "지장없", "틀림없")
PRODUCT_NAME_TERMS = ("개발자", "개발 보드", "개발보드", "개발 키트", "개발키트")
# '비상업적', '비판매용' 처럼 '비' 가 붙은 부정 (앞 글자가 한글이면 '대비판매' 같은 일반 단어이므로 제외)
_NON_PREFIX = re.compile(r"(?<![가-힣])비\s*(상업|판매|유통|양산|납품)")
# "판매 안 해", "유통 안함" 같은 띄어 쓴 부정 (예: '안전'의 '안'은 걸리지 않도록 뒤 동사까지 확인)
_SPACED_NEGATION = re.compile(r"(?:^|\s)안\s*(?:하|해|함|할|합|돼|됩|팔|판|쓰|씀|쓸|넣|들여)")

QC_KEYWORDS = ("품질검사", "품질 검사", "규격 확인", "성능 검사")
QC_LIMIT_RADIO = 10  # kc_exemption.txt 1번: 시험·품질검사 목적 10대 이하
ORIGIN_CERT_KEYWORDS = ("원산지증명서", "원산지 증명서", "원산지증명", "c/o")

COMMERCIAL_NOTICE = (
    "입력에서 판매·상업 목적이 확인되어 연구개발 면제 사유서 초안을 작성하지 않았습니다. "
    "면제 대상 여부는 관세사 또는 소관 기관에 확인하세요."
)

# 품목표에 수량 한도가 없는 품목(limit_quantity=0)에 쓰는 안내 문구 (확인이 필요한 사항 / 수량 한도 검증 / 사유서 말미 공통)
UNREGISTERED_LIMIT_MESSAGE = "법정 수량 한도는 규정되어 있지 않습니다. 요건 면제 충족 여부 확인 후 진행해야 됩니다."

DEFAULT_UNIT_PRICE_KRW = 1_000_000  # 단가 미입력 시 '가정값'. 보고서에 가정임을 반드시 표시한다.


# ── 1) 수량 추출 ────────────────────────────────────────────────────────────
# '1개당 5000원', '1개에 3000원' 처럼 단가 기준 수량(1개)은 수량으로 보지 않는다. '2개월', '3개국', '3개년', '3개소'도 제외.
_QTY_RE = re.compile(r"(\d[\d,]*)\s*(?:대|개|박스|세트|pcs|ea|장|매)(?![a-z]|월|발|당|국|년|소|에\s*\d)", re.IGNORECASE)


def find_quantities(query: str) -> list:
    """질문에서 발견한 수량 후보(중복 제거, 등장 순서)."""
    found = []
    for match in _QTY_RE.finditer(str(query or "")):
        value = int(match.group(1).replace(",", ""))
        if value > 0 and value not in found:
            found.append(value)
    return found


def extract_quantity(query: str):
    """첫 번째 수량. 찾지 못하면 None (예전처럼 임의의 숫자를 수량으로 쓰거나 1로 가정하지 않는다)."""
    quantities = find_quantities(query)
    return quantities[0] if quantities else None


# ── 1) 단가 추출 ────────────────────────────────────────────────────────────
# 금액 토큰: [숫자][억|만|천|백] 반복 + 마지막 순수 숫자 + '원'   예) 5천원, 1만5천원, 2천만원, 12,000원, 만원
_KRW_RE = re.compile(
    r"(?P<amt>(?:(?:\d[\d,]*(?:\.\d+)?)?\s*[억만천백]\s*)*(?:\d[\d,]*(?:\.\d+)?)?)\s*원"
)
_FOREIGN_RE = re.compile(
    r"(?:(?:US\$|\$|€|¥|￥|USD|EUR|CNY|RMB|JPY)\s*\d)"
    r"|(?:\d[\d,.]*\s*[만천]?\s*(?:달러|위안|엔|유로|불|USD|EUR|CNY|RMB|JPY)(?![가-힣]))",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"(?:(\d[\d,]*(?:\.\d+)?)\s*)?([억만천백])|(\d[\d,]*(?:\.\d+)?)")
_UNIT_MULT = {"억": 100_000_000, "만": 10_000, "천": 1_000, "백": 100}
_BUDGET_BEFORE = re.compile(r"예산|한도")
_BUDGET_AFTER = re.compile(r"\s*(?:이내|이하|까지)")
_UNIT_MARKERS = ("개당", "대당", "세트당", "박스당", "단가", "한 개", "한개", "1개", "1대", "개에", "대에", "each", "per")
_TOTAL_MARKERS = ("총액", "총", "합계", "전체", "합해", "합쳐", "전부")


def _parse_krw_amount(amount: str) -> int:
    """'1억 2천만' → 120000000, '5천' → 5000, '12,000' → 12000, '만' → 10000."""
    total = 0.0
    small = 0.0  # 천·백 단위 누적 (다음에 만/억이 오면 그 계수가 된다)
    for match in _TOKEN_RE.finditer(amount):
        num_text, unit, plain = match.group(1), match.group(2), match.group(3)
        if unit in ("천", "백"):
            small += (float(num_text.replace(",", "")) if num_text else 1.0) * _UNIT_MULT[unit]
        elif unit in ("만", "억"):
            coefficient = small + (float(num_text.replace(",", "")) if num_text else 0.0)
            total += (coefficient or 1.0) * _UNIT_MULT[unit]
            small = 0.0
        elif plain:
            small += float(plain.replace(",", ""))
    return int(round(total + small))


def _price_kind(context: str) -> str:
    """금액 바로 앞 문맥에서 '단가'인지 '총액'인지 판별. 더 가까운(뒤쪽) 표지가 우선."""
    lowered = context.lower()
    unit_pos = max((lowered.rfind(m) for m in _UNIT_MARKERS), default=-1)
    total_pos = max((lowered.rfind(m) for m in _TOTAL_MARKERS), default=-1)
    if unit_pos < 0 and total_pos < 0:
        return "unspecified"
    return "unit" if unit_pos >= total_pos else "total"


def extract_unit_price(query: str, quantity=None) -> dict:
    """질문에서 원화 단가를 추출한다.

    반환
      {"status": "found", "unit_krw": int, "basis": "unit" | "total" | "unspecified"}
          basis=total       : 총액을 수량으로 나눠 단가로 환산함
          basis=unspecified : '개당/총' 표지가 없어 단가로 해석함 (보고서에 경고)
      {"status": "foreign"} : 외화 금액만 있음 (환율 적용 필요 → 원화 단가를 다시 입력받아야 함)
      {"status": "missing"} : 금액 없음
    """
    text = str(query or "")
    candidates = []
    for match in _KRW_RE.finditer(text):
        amount = match.group("amt").strip()
        if not amount:
            continue  # '원본', '원래' 등의 '원'
        value = _parse_krw_amount(amount)
        if value <= 0:
            continue
        before = text[max(0, match.start() - 12):match.start()]
        after = text[match.end():match.end() + 6]
        if _BUDGET_BEFORE.search(before) or _BUDGET_AFTER.match(after):
            continue  # 예산·상한은 실제 단가가 아니다
        candidates.append((value, _price_kind(before)))

    if not candidates:
        return {"status": "foreign"} if _FOREIGN_RE.search(text) else {"status": "missing"}

    chosen = (
        next((c for c in candidates if c[1] == "unit"), None)
        or next((c for c in candidates if c[1] == "total"), None)
        or candidates[0]
    )
    value, kind = chosen
    if kind == "total":
        if not quantity:
            return {"status": "missing"}  # 총액인데 수량을 모르면 단가를 구할 수 없다
        return {"status": "found", "unit_krw": int(round(value / quantity)), "basis": "total"}
    return {"status": "found", "unit_krw": value, "basis": kind}


# ── 2) 수입 목적 ────────────────────────────────────────────────────────────
def _mentioned(text: str, keywords) -> bool:
    for kw in keywords:
        start = 0
        while True:
            idx = text.find(kw, start)
            if idx < 0:
                break
            after = text[idx + len(kw): idx + len(kw) + NEGATION_WINDOW]
            after = re.split(r"[,.!?;\n]", after, maxsplit=1)[0]
            for phrase in NEUTRAL_PHRASES:
                after = after.replace(phrase, "")
            negated = any(marker in after for marker in NEGATION_MARKERS) or bool(_SPACED_NEGATION.search(after))
            if not negated:
                return True
            start = idx + len(kw)
    return False


def detect_purpose(query: str) -> str:
    """'commercial' | 'research' | 'unknown'. unknown 이면 보고서는 R&D 목적을 단정하지 않는다."""
    text = str(query or "").lower()
    for term in PRODUCT_NAME_TERMS:
        text = text.replace(term, " ")
    text = _NON_PREFIX.sub(r"\1 아님", text)
    if _mentioned(text, COMMERCIAL_KEYWORDS):
        return "commercial"
    if _mentioned(text, RESEARCH_KEYWORDS):
        return "research"
    return "unknown"


def is_qc_purpose(query: str) -> bool:
    """시험·품질검사 목적(전파법 10대 이하 기준)을 명시했는지."""
    text = str(query or "").lower()
    return any(kw in text for kw in QC_KEYWORDS)


def mentions_origin_cert(query: str) -> bool:
    """원산지증명서(C/O)를 갖고 있다고 말했는지 ('원산지증명서 없어'는 False)."""
    return _mentioned(str(query or "").lower(), ORIGIN_CERT_KEYWORDS)


# ── 3) 세액 계산 ────────────────────────────────────────────────────────────
def calculate_estimated_tax(rate_pct: float, quantity: int, unit_price_krw: int) -> dict:
    """단일 세율 기준 예상 관세·부가세. 부가세 과세표준 = 과세가격 + 관세."""
    total_krw = unit_price_krw * quantity
    estimated_duty = int(total_krw * (rate_pct / 100.0))
    estimated_vat = int((total_krw + estimated_duty) * 0.10)
    return {
        "rate_pct": rate_pct,
        "applied_rate": f"{rate_pct:g}%",
        "estimated_duty": estimated_duty,
        "estimated_vat": estimated_vat,
        "total_tax": estimated_duty + estimated_vat,
    }


def build_tax_scenarios(base_rate_val: float, fta_rate_val: float, quantity: int,
                        unit_price_krw: int, fta_applied: bool = False) -> dict:
    """기본세율 시나리오와 FTA 시나리오를 함께 계산한다.

    FTA 협정세율은 원산지증명서(C/O)와 협정별 원산지 결정 기준 충족이 전제이므로,
    사용자가 C/O 보유를 밝힌 경우(fta_applied=True)에만 'applied' 로 쓰고 기본은 기본세율이다.
    """
    scenarios = {
        "unit_krw": unit_price_krw,
        "quantity": quantity,
        "total_krw": unit_price_krw * quantity,
        "base": calculate_estimated_tax(base_rate_val, quantity, unit_price_krw),
        "fta": calculate_estimated_tax(fta_rate_val, quantity, unit_price_krw),
        "applied_key": "fta" if fta_applied else "base",
    }
    scenarios["applied"] = scenarios[scenarios["applied_key"]]
    return scenarios


# ── 4) 수량 한도 검증 ───────────────────────────────────────────────────────
def check_quantity(quantity: int, limit_quantity: int, law_type: str = "", qc_purpose: bool = False) -> dict:
    """품목표 limit_quantity(0 = 미등록)와 요청 수량을 대조한다.

    state: "within" 한도 이내 | "exceeded" 초과 | "unregistered" 한도 미등록(검증 불가)
    """
    limit = limit_quantity if limit_quantity and limit_quantity > 0 else None
    basis = "품목표 등록 한도"
    if law_type in ("전파법", "복합") and qc_purpose:  # 복합 품목은 전파법 부분의 10대 기준이 함께 적용된다
        limit = QC_LIMIT_RADIO if limit is None else min(limit, QC_LIMIT_RADIO)
        basis = "시험·품질검사 목적 기준"

    if limit is None:
        return {
            "state": "unregistered", "limit": None,
            "message": UNREGISTERED_LIMIT_MESSAGE,
        }
    if quantity <= limit:
        return {
            "state": "within", "limit": limit,
            "message": f"요청 수량({quantity:,}대)은 {basis} {limit:,}대 이하로 한도 이내입니다.",
        }
    return {
        "state": "exceeded", "limit": limit,
        "message": f"요청 수량({quantity:,}대)이 {basis} {limit:,}대를 초과하여 면제 사유서 초안을 작성하지 않았습니다. "
                   "수량 조정 또는 소관 기관 확인이 필요합니다.",
    }
