"""품목표(CSV) 로더 (tariff_catalog.py 수정본)

변경점: 법령 근거와 품목 라벨 분리.
  - law_title 은 그대로 보존(정합성 점검용)하고, 사유서에 쓰는 law_basis(법령 근거)와 item_label(품목 라벨)을 함께 돌려준다.
  - CSV 에 선택 열 law_basis / item_label 을 추가하면 그 값을 쓰고, 없으면 law_title 의 마지막 ' - ' 기준으로 자동 분리한다.
    → 기존 CSV 를 고치지 않아도 동작한다.
"""
import csv
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT_DIR / "data" / "tariff_rates.csv"

REQUIRED_COLUMNS = (
    "hsk_code", "item_name", "law_type", "law_title",
    "limit_quantity", "basic_rate", "fta_rate",
)
HSK_LENGTH = 10

_cache = {"key": None, "items": None}


class CatalogError(Exception):
    """품목표(CSV) 형식 오류."""


def normalize_code(code) -> str:
    return re.sub(r"\D", "", str(code or ""))


def split_law_title(raw_title: str):
    """'법령 근거 - 품목 라벨' → (법령 근거, 품목 라벨). ' - ' 가 없으면 라벨은 빈 문자열."""
    text = str(raw_title or "").strip()
    if " - " in text:
        basis, label = text.rsplit(" - ", 1)
        return basis.strip(), label.strip()
    return text, ""


def normalize_law_basis(text: str) -> str:
    """사유서 제목·본문에 넣기 좋게 법령 근거 표기를 다듬는다 (조문 번호·제목은 건드리지 않는다).

      - 앞머리 '관련 법령:' 제거
      - 중간의 ', 관련 법령 2:' 같은 번호 머리말을 ' 및 ' 로 연결
      - '면제)및' → '면제) 및', '평가)「전파법」' → '평가), 「전파법」'
    """
    out = str(text or "").strip()
    out = re.sub(r"^관련\s*법령\s*\d*\s*:\s*", "", out)
    out = re.sub(r"\s*,?\s*관련\s*법령\s*\d+\s*:\s*", " 및 ", out)
    out = re.sub(r"\)(?=[및와과])", ") ", out)
    out = re.sub(r"(\)|」)\s*(?=「)", r"\1, ", out)
    return re.sub(r"\s{2,}", " ", out).strip()


def _to_rate(value, field: str, line_no: int) -> float:
    text = str(value or "").replace("%", "").strip()
    if not text:
        raise CatalogError(f"{line_no}행: {field} 값이 비어 있습니다.")
    try:
        rate = float(text)
    except ValueError:
        raise CatalogError(f"{line_no}행: {field} 값이 숫자가 아닙니다 ({value!r}).") from None
    if not 0 <= rate <= 100:
        raise CatalogError(f"{line_no}행: {field} 값이 0~100 범위를 벗어났습니다 ({rate}).")
    return rate


def _to_limit(value, line_no: int) -> int:
    text = str(value or "").replace(",", "").strip()
    if not text:
        return 0
    try:
        limit = int(text)
    except ValueError:
        raise CatalogError(f"{line_no}행: limit_quantity 값이 정수가 아닙니다 ({value!r}).") from None
    if limit < 0:
        raise CatalogError(f"{line_no}행: limit_quantity 값이 음수입니다 ({limit}).")
    return limit


def _load(path: Path) -> dict:
    if not path.exists():
        raise CatalogError(f"품목표 파일이 없습니다: {path}")

    items = {}
    with open(path, mode="r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise CatalogError(f"필수 열이 없습니다: {', '.join(missing)}")

        for line_no, row in enumerate(reader, start=2):
            values = [v for k, v in row.items() if k is not None]
            if not any((v or "").strip() for v in values):
                continue

            code = normalize_code(row.get("hsk_code"))
            if len(code) != HSK_LENGTH:
                raise CatalogError(
                    f"{line_no}행: hsk_code는 숫자 {HSK_LENGTH}자리여야 합니다 ({row.get('hsk_code')!r})."
                )
            if code in items:
                raise CatalogError(f"{line_no}행: 중복된 hsk_code 입니다 ({code}).")

            name = (row.get("item_name") or "").strip()
            if not name:
                raise CatalogError(f"{line_no}행: item_name이 비어 있습니다.")

            raw_title = (row.get("law_title") or "").strip()
            derived_basis, derived_label = split_law_title(raw_title)
            # 선택 열: CSV 에 law_basis / item_label 열이 있으면 그 값을 우선하고, 없으면 law_title 에서 나눈다
            law_basis = normalize_law_basis((row.get("law_basis") or "").strip() or derived_basis)
            item_label = (row.get("item_label") or "").strip() or derived_label

            items[code] = {
                "hsk_code": code,
                "item_name": name,
                "law_type": (row.get("law_type") or "").strip(),
                "law_title": raw_title,
                "law_basis": law_basis,
                "item_label": item_label,
                "limit_quantity": _to_limit(row.get("limit_quantity"), line_no),
                "basic_rate_val": _to_rate(row.get("basic_rate"), "basic_rate", line_no),
                "fta_rate_val": _to_rate(row.get("fta_rate"), "fta_rate", line_no),
            }

    if not items:
        raise CatalogError("품목표에 등록된 품목이 하나도 없습니다.")
    return items


def get_catalog() -> dict:
    path = Path(CSV_PATH)
    try:
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        key = None

    if key is not None and _cache["key"] == key:
        return _cache["items"]

    items = _load(path)
    _cache["key"], _cache["items"] = key, items
    return items


def resolve_code(raw_code, allow_unique_prefix: bool = False, catalog: dict = None):
    items = catalog if catalog is not None else get_catalog()
    code = normalize_code(raw_code)
    if not code:
        return None, None

    if code in items:
        return items[code], "exact"

    if allow_unique_prefix and 6 <= len(code) < HSK_LENGTH:
        matches = [entry for full, entry in items.items() if full.startswith(code)]
        if len(matches) == 1:
            return matches[0], "prefix"

    return None, None


def lookup_tariff_strict(hsk_code: str) -> dict:
    try:
        entry, _ = resolve_code(hsk_code, allow_unique_prefix=False)
    except CatalogError as e:
        return {"status": "error", "source": "catalog_error", "message": f"관세율표(CSV) 오류: {e}"}

    if entry is None:
        return {
            "status": "not_found",
            "source": "local_csv",
            "message": f"관세율표에 없는 HSK 코드입니다: {hsk_code}",
        }

    return {
        "status": "success",
        "source": "local_csv",
        "item_name": entry["item_name"],
        "law_type": entry["law_type"],
        "law_title": entry["law_title"],
        "law_basis": entry["law_basis"],
        "item_label": entry["item_label"],
        "limit_quantity": entry["limit_quantity"],
        "hsk_code": entry["hsk_code"],
        "base_rate": f"{entry['basic_rate_val']:g}%",
        "base_rate_val": entry["basic_rate_val"],
        "fta_rate": f"{entry['fta_rate_val']:g}%",
        "fta_rate_val": entry["fta_rate_val"],
        "vat_rate": "10%",
    }
