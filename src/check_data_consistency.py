"""품목표(data/tariff_rates.csv) ↔ 규정 원문(data/kc_exemption.txt) 정합성 점검 (신규 파일)

점검 항목
 1) HSK 10자리 코드 : 규정 원문에 나온 코드가 품목표와 다른지 (앞 6자리는 같은데 10자리가 다르면 ERROR)
 2) 법령 조문 번호   : 같은 조문 제목(예: '안전인증의 면제')인데 품목표와 원문의 '제N조' 가 다른지 (ERROR)
 3) 수량 한도        : 품목표 limit_quantity 와 원문의 'N대 이하' 가 어긋나는지 (WARN)
 4) 누락             : 품목표 품목이 원문에 전혀 없는지 (WARN)

이 스크립트는 '어느 쪽이 맞는지' 판단하지 않는다. 조문 번호는 법령 원문(국가법령정보센터)으로 직접 확인해 CSV 또는 원문 한쪽을 고쳐야 한다.

사용:  python src/check_data_consistency.py      (ERROR 가 있으면 종료코드 1)
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_vector_store import DATA_PATH, extract_hsk_codes, parse_sections  # noqa: E402
from tariff_catalog import CatalogError, get_catalog  # noqa: E402

_ARTICLE_RE = re.compile(r"제\s*(\d+)\s*조(?:\s*의\s*(\d+))?\s*\(([^)]+)\)")
_LIMIT_RE = re.compile(r"(\d[\d,]*)\s*대\s*이하")
_LAW_KEYWORD = {"전파법": ("전파법",), "전안법": ("전기용품 및 생활용품",), "복합": ("전파법", "전기용품 및 생활용품")}


def _articles(text: str) -> dict:
    """{조문 제목: {'6', '58-3', …}}"""
    found = {}
    for number, sub, title in _ARTICLE_RE.findall(text):
        found.setdefault(title.strip(), set()).add(f"{number}-{sub}" if sub else number)
    return found


def _label(number: str) -> str:
    """'6' → '제6조', '58-3' → '제58조의3'"""
    return f"제{number.replace('-', '조의')}" if "-" in number else f"제{number}조"


def _related_sections(entry: dict, sections: list) -> list:
    """품목 코드가 나오는 조항 + (코드 없는) 같은 법령의 일반 조항."""
    code6 = entry["hsk_code"][:6]
    specific = [s for s in sections if code6 in f"{s['title']}\n{s['body']}"]
    keywords = _LAW_KEYWORD.get(entry["law_type"], ())
    general = [
        s for s in sections
        if not extract_hsk_codes(f"{s['title']}\n{s['body']}")
        and any(k in s["body"] for k in keywords)
        and s not in specific
    ]
    return specific + general


def run_checks(catalog: dict = None, sections: list = None) -> list:
    """[(level, message)] — level 은 'ERROR' | 'WARN' | 'INFO'."""
    findings = []
    if catalog is None:
        try:
            catalog = get_catalog()
        except CatalogError as exc:
            return [("ERROR", f"품목표(CSV) 오류: {exc}")]
    if sections is None:
        if not Path(DATA_PATH).exists():
            return [("ERROR", f"규정 원문 파일이 없습니다: {DATA_PATH}")]
        sections = parse_sections(Path(DATA_PATH).read_text(encoding="utf-8"))

    # 1) 원문 속 HSK 코드가 품목표와 일치하는지
    for section in sections:
        for code in extract_hsk_codes(f"{section['title']}\n{section['body']}"):
            if code in catalog:
                continue
            same6 = [c for c in catalog if c[:6] == code[:6]]
            if same6:
                findings.append((
                    "ERROR",
                    f"§{section['number']} HSK {code}: 품목표는 {same6[0]} ({catalog[same6[0]]['item_name']}) — 앞 6자리는 같고 10자리가 다릅니다.",
                ))
            else:
                findings.append(("WARN", f"§{section['number']} HSK {code}: 품목표에 없는 코드입니다."))

    for code, entry in catalog.items():
        related = _related_sections(entry, sections)
        specific = [s for s in related if code[:6] in f"{s['title']}\n{s['body']}"]
        if not specific:
            findings.append(("WARN", f"{code} ({entry['item_name']}): 규정 원문에 이 품목 조항이 없습니다."))

        # 2) 조문 번호
        text_articles = {}
        for section in related:
            for title, numbers in _articles(f"{section['title']}\n{section['body']}").items():
                text_articles.setdefault(title, set()).update(numbers)
        for number, sub, title in _ARTICLE_RE.findall(entry["law_title"]):
            csv_number = f"{number}-{sub}" if sub else number
            theirs = text_articles.get(title.strip())
            if theirs and csv_number not in theirs:
                shown = ", ".join(_label(n) for n in sorted(theirs))
                findings.append((
                    "ERROR",
                    f"{code} ({entry['item_name']}): '{title.strip()}' 조문이 품목표는 {_label(csv_number)}, 규정 원문은 {shown} 입니다. 법령 원문으로 확인 필요.",
                ))

        # 3) 수량 한도
        text_limits = set()
        for section in (specific or related):
            text_limits.update(int(n.replace(",", "")) for n in _LIMIT_RE.findall(section["body"]))
        limit = entry["limit_quantity"]
        if limit > 0 and text_limits and limit not in text_limits:
            findings.append(("WARN", f"{code} ({entry['item_name']}): 품목표 한도 {limit:,}대, 규정 원문 {sorted(text_limits)}대 이하."))
        if limit == 0 and specific:
            specific_limits = set()
            for section in specific:
                specific_limits.update(int(n.replace(",", "")) for n in _LIMIT_RE.findall(section["body"]))
            if specific_limits:
                findings.append(("WARN", f"{code} ({entry['item_name']}): 품목표 한도는 미등록(0)인데 원문에는 {sorted(specific_limits)}대 이하가 있습니다."))

    if not findings:
        findings.append(("INFO", "불일치가 발견되지 않았습니다."))
    return findings


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    findings = run_checks()
    for level, message in findings:
        print(f"[{level}] {message}")
    errors = sum(1 for level, _ in findings if level == "ERROR")
    print(f"\nERROR {errors}건 / WARN {sum(1 for l, _ in findings if l == 'WARN')}건")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
