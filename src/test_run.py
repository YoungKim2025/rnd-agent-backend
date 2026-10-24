"""파이프라인 통합 실행 점검 (test_run.py 대체본)

주의: 실제 Gemini API를 호출합니다 (질문 1건당 최대 3회). 성공 케이스는 output/ 에 md/docx를 만듭니다.
      LLM 응답에 따라 기대값과 다를 수 있으며, 그 경우 [CHECK]로 표시만 합니다.

사용:
    python src/test_run.py          # 기본 케이스 1건
    python src/test_run.py --all    # 전체 케이스
"""
import argparse
import sys
import traceback
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # Windows 콘솔(cp949)에서 한글 출력 오류 방지
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

# (이름, 질문, 기대 status, 기대 stage)
CASES = [
    ("통관 정상", "연구개발 목적으로 리튬이온 배터리 프로토타입 25대를 수입하려고 해.", "success", "done"),
    ("상업 목적 차단", "판매할 목적으로 태블릿 PC 50대를 수입하려고 해.", "error", "report"),
    ("지원 품목 밖", "프랑스산 와인 10병 수입할 때 관세가 얼마야?", "error", "hsk"),
    ("통관과 무관", "오늘 점심 뭐 먹을까?", "unsupported", "router"),
    ("연구실 안전", "연구실 소화기 점검 주기가 어떻게 돼?", "unsupported", "router"),
]


def run_case(run_pipeline, name, query, exp_status, exp_stage) -> bool:
    print(f"\n=== [{name}] ===")
    print(f"질문 : {query}")
    try:
        result = run_pipeline(query)
    except Exception:
        print("[오류] run_pipeline에서 예외가 밖으로 나왔습니다 (설계상 발생하면 안 됨)")
        traceback.print_exc()
        return False

    status, stage = result.get("status"), result.get("stage")
    route = result.get("route") or {}
    inference = result.get("inference") or {}
    print(f"결과 : status={status}, stage={stage}")
    if route:
        print(f"의도 : {route.get('category')} (신뢰도 {route.get('confidence')}, 폴백 {route.get('fallback_used')})")
    if inference:
        print(f"HSK  : {inference.get('hsk_code') or '-'} / {inference.get('item_name') or '-'}")
    if status == "success":
        print(f"MD   : {result.get('md_path')}")
        print(f"DOCX : {result.get('docx_path')}")
    else:
        print(f"메시지: {result.get('message')}")

    ok = (status == exp_status and stage == exp_stage)
    print("판정 : OK" if ok else f"판정 : [CHECK] 기대값은 status={exp_status}, stage={exp_stage}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="전체 케이스 실행 (기본: 첫 케이스만)")
    args = parser.parse_args()

    print("1. 모듈 임포트 시도...")
    try:
        from pipeline_runner import run_pipeline
        from main_pipeline import ROOT_DIR, OUTPUT_DIR, CHROMA_DIR
    except Exception:
        print("[오류] 임포트 실패")
        traceback.print_exc()
        return 2
    print(f"   - 루트 디렉토리 : {ROOT_DIR}")
    print(f"   - 출력 디렉토리 : {OUTPUT_DIR}")
    print(f"   - 벡터스토어    : {CHROMA_DIR} (존재 여부: {CHROMA_DIR.exists()})")

    print("\n2. 파이프라인 실행...")
    cases = CASES if args.all else CASES[:1]
    results = [run_case(run_pipeline, *case) for case in cases]

    print(f"\n3. 요약: {sum(results)}/{len(results)} 건 기대값 일치")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
