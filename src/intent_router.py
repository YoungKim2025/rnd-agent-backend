import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

CUSTOMS = "customs_and_certification"
LAB_SAFETY = "lab_safety"
GENERAL = "general_chat"
UNKNOWN = "unknown"

VALID_CATEGORIES = (CUSTOMS, LAB_SAFETY, GENERAL)
CONFIDENCE_THRESHOLD = 0.6
MAX_INPUT_CHARS = 1000

FALLBACK_KEYWORDS = {
    CUSTOMS: ("통관", "관세", "hs code", "hsk", "kc", "인증", "면제", "수입", "fta", "전파법", "전안법", "세관"),
    LAB_SAFETY: ("연구실안전", "연구실 안전", "안전점검", "안전 점검", "화재", "유해위험", "폐기물", "환경 규제")
}

SYSTEM_PROMPT = """당신은 R&D용 전자기기 수입 통관 에이전트의 '최전단 라우터(Intent Router)'입니다.
사용자의 질문을 읽고 다음 3가지 카테고리 중 정확히 하나로 분류하세요.

[카테고리]
1. customs_and_certification: 해외 부품 수입, 통관, HSK 코드, 관세율, KC 인증 및 면제, 전파법/전안법 관련 질문.
2. lab_safety: 연구실 안전, 화재 점검, 소화기, 유해위험기계 점검, 폐기물 처리 관련 질문.
3. general_chat: 인사, 서비스 소개, 위 두 가지에 속하지 않는 일상 대화 또는 코딩 질문 등.

[응답 포맷]
반드시 아래 JSON 형식으로만 응답하세요. 다른 설명은 절대 추가하지 마세요.
{
    "category": "분류된 카테고리명",
    "confidence": 0.0 ~ 1.0 사이의 숫자,
    "reason": "분류에 대한 간략한 한 문장 이유"
}
"""

@dataclass
class RouteResult:
    category: str
    confidence: float
    reason: str
    fallback_used: bool = False

class IntentRouter:
    def __init__(self, confidence_threshold=CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold
        # 💡 config.py의 중앙 클라이언트(타임아웃 적용) 사용
        self.client = config.make_client()

    def _keyword_fallback(self, text: str) -> str:
        lowered = text.lower()
        scores = {cat: sum(kw in lowered for kw in kws) for cat, kws in FALLBACK_KEYWORDS.items()}
        top = max(scores.values()) if scores else 0
        winners = [cat for cat, score in scores.items() if score == top]
        if top == 0 or len(winners) > 1:
            return UNKNOWN
        return winners[0]

    def route(self, user_input: str) -> RouteResult:
        text = str(user_input).strip()[:MAX_INPUT_CHARS]
        if not text:
            return RouteResult(UNKNOWN, 0.0, "빈 입력입니다.")

        try:
            # 💡 503/429 등 일시적 오류에 대한 중앙화된 3회 재시도 적용
            response = config.with_retry(
                lambda: self.client.models.generate_content(
                    model=config.MODEL_NAME,  # 환경변수 모델명 동적 적용
                    contents=f"{SYSTEM_PROMPT}\n\n[사용자 질문]\n{text}"
                )
            )
            
            raw_text = response.text.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw_text)
            
            category = data.get("category", UNKNOWN)
            confidence = float(data.get("confidence", 0.0))
            reason = data.get("reason", "")
            
            if category not in VALID_CATEGORIES:
                category = UNKNOWN
      
        except Exception as e:
            fallback_cat = self._keyword_fallback(text)
            error_info = config.format_api_error(e, task_name="의도 분류")
            
            return RouteResult(
                category=fallback_cat, 
                confidence=self.confidence_threshold if fallback_cat != UNKNOWN else 0.5,
                # 💡 JSON 에러 원문 대신 정제된 메시지 삽입
                reason=f"규칙 기반 대체 발동: {error_info['message']}",
                fallback_used=True,
                # 필요시 error_kind를 RouteResult 데이터 클래스에 추가하거나 활용 가능
            )

        if confidence < self.confidence_threshold:
            return RouteResult(
                category=UNKNOWN, 
                confidence=confidence,
                reason=f"신뢰도 부족 ({confidence} < {self.confidence_threshold}): {reason}"
            )
            
        return RouteResult(category, confidence, reason)