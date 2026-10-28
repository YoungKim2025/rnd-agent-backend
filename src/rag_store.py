"""벡터 스토어 검색 (rag_store.py 수정본) — 메타데이터 직접 조회 적용"""
import json
import re
from pathlib import Path

from langchain_core.documents import Document
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

    # 임베딩 API 동적 호출을 방지하기 위해 정적 서명으로 메타데이터만 검증
    # (구축 시 사용했던 backend와 model명을 고정 입력합니다)
    check_store_meta({"backend": "gemini", "model": "models/embedding-001"})

    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma
        
    # 벡터 검색을 하지 않고 메타데이터 필터링(.get)만 사용하므로 embedding_function을 분리
    from langchain_core.embeddings import FakeEmbeddings
    # Chroma는 내부적으로 embedding_function이 없으면 기본 모델을 다운로드하려 시도할 수 있으므로, 빈 Fake 껍데기를 주입합니다.
    return Chroma(persist_directory=str(CHROMA_DIR), embedding_function=FakeEmbeddings(size=1))


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
    """임베딩 없이 메타데이터 필터링만으로 규정 조항(Document)을 직접 가져옵니다.
    (호환성을 위해 query 파라미터는 유지하되 내부에서는 사용하지 않습니다.)
    """
    if store is None:
        store = get_store()
        
    docs, seen = [], set()

    def add_from_get(flt: dict):
        results = store.get(where=flt)
        for i in range(len(results.get("documents", []))):
            content = results["documents"][i]
            meta = results["metadatas"][i]
            key = (meta.get("section"), meta.get("part"))
            if key not in seen:
                seen.add(key)
                docs.append(Document(page_content=content, metadata=meta))

    # 1. 품목 전용 조항 (해당 HSK 코드가 명시된 조항만)
    add_from_get({"$and": [{"category": CATEGORY}, {hsk6_flag(hsk_code): True}]})

    # 2. 법령 일반 조항 (타 품목 전용 조항이 섞이지 않도록 scope=general 만)
    add_from_get({"$and": [
        {"category": CATEGORY},
        {"scope": "general"},
        {"law_type": {"$in": allowed_law_types(law_type_kr)}},
    ]})

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