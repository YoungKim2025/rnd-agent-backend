import os
import google.generativeai as genai
from dotenv import load_dotenv

# 1. .env 파일에서 환경변수 로드
load_dotenv()

# 2. Gemini API 키 설정 (기획서에 명시된 gemini-3.5-flash 모델 사용 준비)
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")

genai.configure(api_key=api_key)

def get_hsk_code(user_query: str) -> str:
    """
    사용자의 자연어 입력을 받아 Gemini LLM을 통해 HSK 10자리 코드를 추론하는 함수
    """
    # 에이전트 기획안에 맞춘 시스템 프롬프트 설계
    system_prompt = """
    당신은 한국 통관 및 관세 실무를 담당하는 AI 관세사입니다.
    사용자가 입력하는 R&D용 전자기기, 하드웨어 부품(특히 제84류, 제85류 중심)의 설명을 분석하여
    가장 적합한 한국 관세율표 기준 HSK 10자리 코드를 추론하세요.

    [분류 참고 데이터 (제84/85류)]
    - 태블릿 PC, 휴대용 자동자료처리기계: 8471300000
    - 무선통신모듈 (Wi-Fi, Bluetooth, 통신 보드 등): 8517629090
    - 마이크로컨트롤러/프로세서 (MCU, 반도체): 8542311000
    - 리튬이온 축전지 (배터리): 8507600000
    - 전원어댑터: 8504409090
    - 접속자가 부착된 케이블/전선: 8544422090

    제약사항: 부가적인 설명이나 인사말을 절대 포함하지 말고, 오직 '10자리 숫자(HSK 코드)'만 출력하세요.
    """
    
    # 모델 인스턴스 생성 (gemini-3.5-flash)
    model = genai.GenerativeModel('gemini-3.5-flash')
    
    # 프롬프트와 사용자 입력을 함께 전달하여 결과 생성
    response = model.generate_content([system_prompt, user_query])
    
    # 공백 및 줄바꿈 제거 후 반환
    return response.text.strip()

# === 테스트 실행 코드 ===
if __name__ == "__main__":
    print("Gemini API HSK Code 추론 테스트를 시작합니다...\n")
    
    # 테스트 1: 통신 모듈 (과거 ESP32-M1 프로젝트 관련 경험 반영)
    test_query_1 = "이번에 ESP32-M1 Reach Out 하드웨어 프로젝트에 쓸 Wi-Fi 듀얼밴드 통신 보드를 수입하려고 해."
    print(f"사용자 입력 1: {test_query_1}")
    print(f"추론된 HSK 코드: {get_hsk_code(test_query_1)}\n")
    
    # 테스트 2: 태블릿 PC (과거 KC 면제 신청 경험 반영)
    test_query_2 = "연구개발 목적으로 중국에서 안드로이드 태블릿 PC 프로토타입 5대를 들여올 예정이야."
    print(f"사용자 입력 2: {test_query_2}")
    print(f"추론된 HSK 코드: {get_hsk_code(test_query_2)}\n")