"""공용 설정 / Gemini 호출 헬퍼 (config.py 정리본)

 * 모델명·타임아웃·재시도 정책을 한 곳에서 관리한다. 환경변수로 덮어쓸 수 있다.
     GEMINI_MODEL            (기본 gemini-3.8-flash)
     GEMINI_EMBEDDING_MODEL  (기본 gemini-embedding-001)
     EMBEDDING_BACKEND       gemini(기본) | hf
     HF_EMBEDDING_MODEL      (hf 선택 시, 기본 jhgan/ko-sroberta-multitask)
     GEMINI_TIMEOUT_S        (기본 30초)

 * 재시도 정책 (with_retry)
     - 서버 오류(5xx, ServerError)·503/UNAVAILABLE·타임아웃: 2초 간격으로 최대 3회 재시도
     - 429(할당량 초과): 재시도하지 않고 즉시 실패 → format_api_error 가 사용자용 안내문으로 바꾼다
 * 클라이언트는 get_client() 로 한 번만 만들어 재사용한다 (질문마다 새로 만들지 않음).
 * 무거운 패키지(google-genai)는 함수 안에서 import 하므로, 순수 로직 테스트에는 필요 없다.
"""
import os
import re
import time
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:  # python-dotenv 미설치: 시스템 환경변수만 사용
    pass

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = ROOT_DIR / "vector_store"
OUTPUT_DIR = ROOT_DIR / "output"

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "gemini").strip().lower()
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "jhgan/ko-sroberta-multitask")
REQUEST_TIMEOUT_S = float(os.getenv("GEMINI_TIMEOUT_S", "30"))
MAX_RETRIES = 3
RETRY_SLEEP_S = 2

_TRANSIENT_CODES = (500, 502, 503, 504)
_TRANSIENT_MARKERS = ("503", "unavailable", "deadline", "timed out", "timeout")
_QUOTA_MARKERS = ("429", "resource_exhausted")


# ── 오류 분류 ───────────────────────────────────────────────────────────────
def is_quota_error(exc: Exception) -> bool:
    """429 / RESOURCE_EXHAUSTED (사용량 한도 초과)."""
    if getattr(exc, "code", None) == 429:
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _QUOTA_MARKERS)


def is_transient(exc: Exception) -> bool:
    """재시도할 만한 일시적 오류인가. 할당량 초과(429)는 재시도해도 소용없으므로 제외한다."""
    if is_quota_error(exc):
        return False
    if type(exc).__name__ == "ServerError":  # google.genai.errors.ServerError (5xx 전반)
        return True
    if getattr(exc, "code", None) in _TRANSIENT_CODES:
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def with_retry(fn, retries: int = MAX_RETRIES):
    """일시적 오류만 RETRY_SLEEP_S 간격으로 재시도하고, 나머지는 즉시 예외를 던진다."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - SDK 예외 계층이 버전마다 달라 폭넓게 받는다
            if is_transient(exc) and attempt < retries - 1:
                time.sleep(RETRY_SLEEP_S)
                continue
            raise


def format_api_error(exc: Exception, task_name: str = "AI 작업") -> dict:
    """예외를 화면에 보여줄 정제된 메시지와 error_kind(config|quota|overloaded|other)로 바꾼다. 원문은 숨긴다."""
    # 0. API 키 미설정은 원인을 그대로 알려준다 (일반 오류로 뭉뚱그리지 않음)
    if isinstance(exc, RuntimeError) and "GEMINI_API_KEY" in str(exc):
        return {"status": "error", "error_kind": "config", "message": str(exc)}

    # 1. 429 사용량 초과 (재시도 없이 즉시 실패)
    if is_quota_error(exc):
        msg = "API 사용량 한도를 초과했습니다. 잠시 후 다시 시도해 주세요."
        match = re.search(r"retry in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", str(exc).lower())
        if match:
            h, m, s = match.groups()
            total_s = int(h or 0) * 3600 + int(m or 0) * 60 + int(float(s or 0))
            if total_s > 3600:
                msg += " (일일 한도일 수 있어 내일 다시 시도해 주세요.)"
            else:
                msg += f" (약 {total_s}초 후 재시도 가능)"
        else:
            msg += " (일일 한도일 수 있습니다.)"
        return {"status": "error", "error_kind": "quota", "message": msg}

    # 2. 서버 혼잡/타임아웃 (재시도 3회가 모두 실패한 경우)
    if is_transient(exc):
        return {"status": "error", "error_kind": "overloaded", "message": "AI 서버가 혼잡합니다. 잠시 후 다시 시도해 주세요."}

    # 3. 그 밖의 오류 (원문 숨김)
    return {"status": "error", "error_kind": "other", "message": f"{task_name} 중 오류가 발생했습니다 ({type(exc).__name__})."}


# ── 클라이언트 ──────────────────────────────────────────────────────────────
_client = None


def make_client():
    """타임아웃이 설정된 새 genai.Client. 키가 없으면 RuntimeError."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=int(REQUEST_TIMEOUT_S * 1000)),  # 밀리초
    )


def get_client():
    """프로세스 안에서 한 번만 만들어 재사용하는 클라이언트."""
    global _client
    if _client is None:
        _client = make_client()
    return _client


def generate_text(prompt: str) -> str:
    """프롬프트 하나를 보내 텍스트를 돌려준다 (타임아웃 + 재시도 적용)."""
    client = get_client()
    response = with_retry(lambda: client.models.generate_content(model=MODEL_NAME, contents=prompt))
    return (response.text or "").strip()
