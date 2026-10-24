"""Phase 1 / Step 1: Gemini API 연동 및 기본 검증

실행 (어느 위치에서든 가능):
    python src/test_gemini.py

종료 코드:
    0 = 성공
    1 = API 호출 실패 (키 오류, 모델명 오류, 네트워크 등)
    2 = API 키 설정 오류 (.env 또는 환경변수에서 키를 찾지 못함)
    3 = 필수 패키지 미설치
"""
import os
import sys
import time
from pathlib import Path

MODEL_NAME = "gemini-3.8-flash"
TEST_MESSAGE = "통관 에이전트 시스템 연결 테스트 중입니다"
KEY_ENV_NAMES = ("GEMINI_API_KEY", "GOOGLE_API_KEY")  # 먼저 발견되는 변수를 사용
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"  # rnd_agent_project/.env


def load_api_key():
    """프로젝트 루트의 .env를 로드한 뒤 (변수명, 키)를 반환한다. 없으면 (None, None)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("[경고] python-dotenv 미설치: .env는 읽지 않고 시스템 환경변수만 확인합니다.")
    else:
        load_dotenv(ENV_PATH)

    for name in KEY_ENV_NAMES:
        value = os.environ.get(name, "").strip().strip('"').strip("'")
        if value:
            return name, value
    return None, None


def mask_key(key):
    """키 원문은 출력하지 않고 앞 4자리와 길이만 보여준다."""
    return f"{key[:4]}***(길이 {len(key)})"


def extract_text(content):
    """응답 content(문자열 또는 블록 리스트)에서 텍스트만 추출한다."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts).strip()
    return str(content).strip()


def main():
    print("=" * 50)
    print("Gemini API 연동 테스트 (Phase 1 / Step 1)")
    print("=" * 50)
    print(f".env 경로 : {ENV_PATH} ({'있음' if ENV_PATH.exists() else '없음'})")

    name, api_key = load_api_key()
    if not api_key:
        print("[오류] API 키를 찾지 못했습니다. .env에 다음 중 하나를 설정하세요:")
        print("       GEMINI_API_KEY=발급받은_키   (또는 GOOGLE_API_KEY=발급받은_키)")
        return 2
    print(f"API 키    : {name} = {mask_key(api_key)}")
    print(f"모델      : {MODEL_NAME}")

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        print("[오류] langchain-google-genai가 설치되어 있지 않습니다.")
        print("       pip install langchain-google-genai python-dotenv")
        return 3

    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=api_key, temperature=0)
    prompt = f'다음 문장을 받았다는 확인 응답을 한국어 한 문장으로만 답하세요: "{TEST_MESSAGE}"'

    print(f"\n요청 문장 : {TEST_MESSAGE}")
    print("Gemini 응답 대기 중...\n")

    start = time.perf_counter()
    try:
        response = llm.invoke(prompt)
    except Exception as e:  # 키/모델명/네트워크 등 원인 확인용으로 폭넓게 처리
        print(f"[실패] {type(e).__name__}: {e}")
        print("점검 항목: (1) API 키 유효성  (2) 모델명 gemini-3.5-flash 사용 가능 여부  (3) 인터넷 연결")
        return 1
    elapsed = time.perf_counter() - start

    text = extract_text(response.content)
    if not text:
        print("[실패] 응답은 받았지만 텍스트가 비어 있습니다.")
        return 1

    print(f"[성공] 응답 시간 {elapsed:.2f}초")
    print(f"Gemini 응답: {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
