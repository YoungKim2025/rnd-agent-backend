"""임베딩 팩토리 (embeddings.py 수정본)

변경점: get_embeddings() 결과를 캐시해 재사용하고, Gemini 클라이언트도 config.get_client() 를 공유한다.
        (질문마다 클라이언트·임베딩 객체를 새로 만들지 않는다)

 * 기본 : Gemini 임베딩 API (이미 쓰고 있는 google-genai + GEMINI_API_KEY 재사용, 추가 설치 없음)
 * 대안 : EMBEDDING_BACKEND=hf  → sentence-transformers 로컬 모델 (오프라인 가능, pip install sentence-transformers)

벡터 DB를 만든 임베딩과 검색 때 임베딩이 다르면 결과가 무의미하므로,
build_vector_store.py 가 vector_store/embedding_meta.json 에 서명(+구조 버전)을 기록하고 rag_store.py 가 대조한다.
"""
from langchain_core.embeddings import Embeddings

import config

_BATCH = 50  # embed_content 1회 요청 상한(100)보다 여유 있게
_cache = {}


class GeminiEmbeddings(Embeddings):
    def __init__(self, model: str = None):
        self.model = model or config.EMBEDDING_MODEL
        self._client = config.get_client()

    def _embed(self, texts, task_type: str):
        from google.genai import types

        vectors = []
        for start in range(0, len(texts), _BATCH):
            batch = texts[start:start + _BATCH]
            response = config.with_retry(
                lambda b=batch: self._client.models.embed_content(
                    model=self.model,
                    contents=b,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
            )
            vectors.extend([list(e.values) for e in response.embeddings])
        return vectors

    def embed_documents(self, texts):
        return self._embed(list(texts), "RETRIEVAL_DOCUMENT")

    def embed_query(self, text):
        return self._embed([text], "RETRIEVAL_QUERY")[0]


def embedding_signature() -> dict:
    if config.EMBEDDING_BACKEND == "hf":
        return {"backend": "hf", "model": config.HF_EMBEDDING_MODEL}
    return {"backend": "gemini", "model": config.EMBEDDING_MODEL}


def _create_embeddings():
    if config.EMBEDDING_BACKEND == "hf":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=config.HF_EMBEDDING_MODEL)
    return GeminiEmbeddings()


def get_embeddings():
    """설정이 같으면 같은 객체를 돌려준다."""
    key = tuple(sorted(embedding_signature().items()))
    if key not in _cache:
        _cache[key] = _create_embeddings()
    return _cache[key]
