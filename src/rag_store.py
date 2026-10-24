"""벡터 스토어 검색 (rag_store.py 수정본) — main_pipeline.py 와 rag_search.py 가 공용으로 사용

검색 순서 (build_vector_store.py 의 메타데이터 규칙과 짝)
 1) 품목 전용 조항 : hsk6_<앞6자리> == True  (해당 품목 HSK가 명시된 조항)
 2) 법령 일반 조항 : scope == "general" 이고 law_type ∈ {품목의 법령, complex, common}
    → HSK 코드가 없는 일반 기준(예: 태블릿 연구개발용 1,500대 이하)과 절차·지침 조항.
       다른 품목의 전용 조항(scope=specific)은 섞이지 않는다.
두 결과를 합쳐 중복을 제거하고 최대 MAX_DOCS 건을 돌려준다.

변경점
 - 구조 버전 확인: embedding_meta.json 의 schema 가 코드(SCHEMA_VERSION)와 다르면 조용히 검색 결과가 줄어드는 대신 오류로 알린다.
 - 벡터 스토어를 한 번만 열어 재사용한다 (DB 파일이 바뀌면 자동으로 다시 연다).
 - 질문 임베딩을 한 번만 계산해 두 검색에 함께 쓴다 (예전에는 2회 호출).
"""
import json
import re
from pathlib import Path

from build_vector_store import SCHEMA_VERSION

ROOT_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT_DIR / "vector_store"
META_NAME = "embedding_meta.json"
CATEGORY = "customs_and_certification"
MAX_DOCS = 4

# 품목표(CSV)의 law_type(한글) → 벡터 스토어의 law_type
LAW_TYPE_MAP = {"전파법": "radio", "전안법": "electrical", "복합": "complex"}


class VectorStoreError(Exception):
    """벡터 스토어가 없거나, 임베딩 서명·구조 버전이 코드와 맞지 않음."""


def allowed_law_types(law_type_kr: str) -> list:
    """일반 조항 검색에서 허용할 law_type 목록."""
    mapped = LAW_TYPE_MAP.get(law_type_kr or "")
    if mapped is None or mapped == "complex":
        return sorted({"radio", "electrical", "complex", "common"})
    return sorted({mapped, "complex", "common"})


def hsk6_flag(hsk_code: str) -> str:
    return f"hsk6_{re.sub(r'[^0-9]', '', str(hsk_code))[:6]}"


def check_store_meta(signature: dict, chroma_dir: Path = None) -> None:
    """임베딩 서명과 구조 버전을 확인한다. 맞지 않으면 VectorStoreError."""
    meta_path = Path(chroma_dir or CHROMA_DIR) / META_NAME
    if not meta_path.exists():
        raise VectorStoreError(
            "벡터 스토어에 서명 파일(embedding_meta.json)이 없습니다. 이전 버전으로 만든 DB이므로 "
            "`python src/build_vector_store.py` 로 다시 구축하세요."
        )
    built = json.loads(meta_path.read_text(encoding="utf-8"))

    if built.get("schema") != SCHEMA_VERSION:
        raise VectorStoreError(
            f"벡터 스토어의 구조 버전이 코드와 다릅니다 (DB: {built.get('schema')}, 코드: {SCHEMA_VERSION}). "
            "`python src/build_vector_store.py` 로 다시 구축하세요."
        )
    built_signature = {key: built.get(key) for key in ("backend", "model")}
    if built_signature != signature:
        raise VectorStoreError(
            f"벡터 스토어는 {built_signature} 로 만들어졌지만 현재 설정은 {signature} 입니다. "
            "EMBEDDING_BACKEND/모델 설정을 맞추거나 build_vector_store.py 로 다시 구축하세요."
        )


def load_vectorstore():
    if not CHROMA_DIR.exists():
        raise VectorStoreError("벡터 스토어가 없습니다. `python src/build_vector_store.py` 를 먼저 실행하세요.")

    from embeddings import embedding_signature, get_embeddings

    check_store_meta(embedding_signature())

    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma
    return Chroma(persist_directory=str(CHROMA_DIR), embedding_function=get_embeddings())


# ── 스토어 캐시: DB 파일(수정 시각)이 바뀌면 자동으로 다시 연다 ───────────────────────
_cache = {"key": None, "store": None}


def _cache_key() -> tuple:
    stamps = []
    for name in ("chroma.sqlite3", META_NAME):
        path = CHROMA_DIR / name
        stamps.append(path.stat().st_mtime_ns if path.exists() else None)
    return tuple(stamps)


def get_store():
    key = _cache_key()
    if _cache["store"] is None or _cache["key"] != key:
        _cache["store"] = load_vectorstore()
        _cache["key"] = key
    return _cache["store"]


# ── 검색 ────────────────────────────────────────────────────────────────────
def retrieve_legal_docs(query: str, hsk_code: str, law_type_kr: str, store=None, embedder=None) -> list:
    """관련 규정 조항(Document) 목록. 아무것도 못 찾으면 빈 리스트.

    store/embedder 를 따로 주지 않으면 캐시된 스토어와 임베딩을 쓰고, 질문 임베딩은 한 번만 계산한다.
    embedder 없이 store 만 주면(테스트·도구용) 예전처럼 텍스트로 검색한다.
    """
    if store is None:
        store = get_store()
        if embedder is None:
            from embeddings import get_embeddings

            embedder = get_embeddings()
    vector = embedder.embed_query(query) if embedder is not None else None

    def search(k: int, flt: dict):
        if vector is not None:
            return store.similarity_search_by_vector(vector, k=k, filter=flt)
        return store.similarity_search(query, k=k, filter=flt)

    docs, seen = [], set()

    def add(results):
        for doc in results:
            key = (doc.metadata.get("section"), doc.metadata.get("part"))
            if key not in seen:
                seen.add(key)
                docs.append(doc)

    # 1. 품목 전용 조항 (해당 HSK 코드가 명시된 조항만)
    add(search(2, {"$and": [{"category": CATEGORY}, {hsk6_flag(hsk_code): True}]}))

    # 2. 법령 일반 조항 (타 품목 전용 조항이 섞이지 않도록 scope=general 만)
    add(search(2, {"$and": [
        {"category": CATEGORY},
        {"scope": "general"},
        {"law_type": {"$in": allowed_law_types(law_type_kr)}},
    ]}))

    return docs[:MAX_DOCS]


def format_context(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def source_labels(docs: list) -> list:
    """보고서 '참조 규정' 표기용: '3. 직류전원장치 및 어댑터 (kc_exemption.txt)'."""
    labels = []
    for doc in docs:
        label = f"{doc.metadata.get('title', '제목 없음')} ({doc.metadata.get('source', 'kc_exemption.txt')})"
        if label not in labels:
            labels.append(label)
    return labels
