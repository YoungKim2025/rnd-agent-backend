"""느린 구간 진단 (신규, 점검용) — 단계별 소요 시간을 출력한다.

    python src/time_check.py

각 단계는 실제 파이프라인이 쓰는 코드를 그대로 호출한다. 어느 단계에서 오래 걸리는지 보면 원인을 알 수 있다.
  1 클라이언트 생성  : 키 누락/패키지 문제
  2 임베딩 1건       : Gemini 임베딩 API 응답 속도 (네트워크/할당량)
  3 벡터 검색        : Chroma 로드 + 검색 (임베딩 1회 포함)
  4 LLM 1건          : Gemini 생성 API 응답 속도
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import config  # noqa: E402


def step(label, fn):
    start = time.time()
    try:
        result = fn()
        print(f"[{time.time() - start:5.1f}초] {label}: OK")
        return result
    except Exception as exc:  # noqa: BLE001
        print(f"[{time.time() - start:5.1f}초] {label}: 실패 - {type(exc).__name__}: {str(exc)[:200]}")
        return None


print(f"모델={config.MODEL_NAME} / 임베딩={config.EMBEDDING_BACKEND}:{config.EMBEDDING_MODEL} / 타임아웃={config.REQUEST_TIMEOUT_S}초")
step("1 클라이언트 생성", config.make_client)

from embeddings import get_embeddings  # noqa: E402

embedder = step("2a 임베딩 객체 생성", get_embeddings)
if embedder:
    step("2b 임베딩 1건", lambda: embedder.embed_query("연구개발 목적 배터리 수입 면제"))

from rag_store import retrieve_legal_docs  # noqa: E402

docs = step("3 벡터 검색(Chroma 로드 포함)", lambda: retrieve_legal_docs("태블릿 PC 연구개발 수량", "8471300000", "복합"))
if docs is not None:
    print(f"        검색된 조항 {len(docs)}건: " + " | ".join(d.metadata.get("title", "")[:25] for d in docs))

step("4 LLM 1건", lambda: config.generate_text("한 단어로 인사해 주세요."))
