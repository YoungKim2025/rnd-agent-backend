import csv
from pathlib import Path

# 프로젝트 루트 경로 기준 data/tariff_rates.csv 경로 설정
ROOT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT_DIR / "data" / "tariff_rates.csv"

def lookup_tariff(hsk_code: str) -> dict:
    """
    로컬 tariff_rates.csv 파일에서 HSK 코드를 조회하여 관세율 정보를 반환합니다.
    """
    if not CSV_PATH.exists():
        return {"error": f"관세율 데이터 파일이 존재하지 않습니다: {CSV_PATH}"}

    # HSK 코드 표준화 (하이픈 및 공백 제거)
    target_code = hsk_code.replace("-", "").replace(".", "").replace(" ", "").strip()

    with open(CSV_PATH, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_code = row.get("hsk_code", "").replace("-", "").replace(".", "").replace(" ", "").strip()
            if csv_code == target_code:
                return {
                    "status": "success",
                    "item_name": row.get("item_name", "알 수 없음"),
                    "hs_code": row.get("hs_code", row.get("hsk_code", ""))[:6],
                    "hsk_code": row.get("hsk_code", hsk_code),
                    "base_rate": row.get("basic_rate", "0") + "%",
                    "wto_rate": row.get("wto_rate", "0") + "%",
                    "fta_rate_us": row.get("fta_us", "0") + "%",
                    "fta_rate_cn": row.get("fta_cn", "0") + "%",
                    "vat_rate": row.get("vat", "10") + "%"
                }
    
    return {"status": "not_found", "message": f"해당 HSK 코드를 데이터베이스에서 찾을 수 없습니다: {hsk_code}"}

def generate_tariff_report(hsk_code: str) -> str:
    """
    확정된 HSK 코드를 바탕으로 예상 관세율표 마크다운 보고서를 생성합니다.
    """
    result = lookup_tariff(hsk_code)
    
    if result.get("status") != "success":
        return f"### [조회 실패]\n{result.get('message')}"

    report = f"""### 📊 예상 관세율표 및 세액 산출 초안

- **품목명**: {result['item_name']}
- **HS Code (6자리)**: {result['hs_code']}
- **HSK Code (10자리)**: {result['hsk_code']}

| 구분 | 적용 세율 | 비고 / 협정 |
| :--- | :---: | :--- |
| **기본 관세율** | {result['base_rate']} | 일반 기본 세율 |
| **WTO 협정세율** | {result['wto_rate']} | WTO 양허세율 |
| **한미 FTA** | {result['fta_rate_us']} | 미국 원산지 적용 시 |
| **한중 FTA** | {result['fta_rate_cn']} | 중국 원산지 적용 시 |
| **부가가치세 (VAT)** | {result['vat_rate']} | (과세가격 + 관세) 기준 부과 |

> **[안내]** 본 예상 관세율표는 로컬 관세율 데이터(`tariff_rates.csv`)를 기반으로 자동 생성되었습니다. 실제 통관 시 원산지증명서 및 품목분류 사전심사 결과에 따라 달라질 수 있습니다.
"""
    return report

if __name__ == "__main__":
    print("=" * 50)
    print("관세율 조회 모듈 테스트")
    print("=" * 50)
    
    # 테스트할 샘플 HSK 코드들 (하이픈 유무 관계없이 조회 가능)
    test_codes = ["8471.30-0000", "8517629090", "8507.60-0000"]
    
    for code in test_codes:
        print(f"\n[조회 요청 HSK]: {code}")
        report = generate_tariff_report(code)
        print(report)
        print("-" * 50)