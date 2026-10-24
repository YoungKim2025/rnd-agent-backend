"""규정 원문(data/kc_exemption.txt) → ChromaDB 벡터 스토어 구축 (build_vector_store.py 대체본)

변경점
 - 글자 수(500자)로 자르던 방식 → '번호 붙은 조항(1. 2. 3. …)' 단위로 분할. 조항이 청크 중간에서 끊기지 않는다.
 - 메타데이터 정정
     law_type  : radio(전파법) | electrical(전안법) | complex(복합) | common(절차·지침 등)
     hsk6_<앞6자리> : True   ← 조항에 나오는 모든 HSK 코드마다 플래그 (한 조항에 여러 품목이 있어도 전부 검색됨)
     예전 방식은 청크당 첫 번째로 발견된 코드 하나만 hsk_prefix 로 달아서, 나머지 품목 조항을 찾지 못했다.
   HSK 코드가 없는 일반 조항(예: 태블릿 1,500대 기준)은 scope=general + law_type 으로 검색된다.
 - scope 메타데이터: HSK 코드가 있는 조항은 specific, 없으면 general. 일반 조항 검색에서 다른 품목의 전용 조항이 섞이지 않게 한다.
 - 구조 버전(SCHEMA_VERSION)을 embedding_meta.json 에 기록한다. 코드와 DB의 버전이 다르면 rag_store 가 재구축을 요구한다.
 - 한국어 임베딩 사용 (embeddings.py)
 - 안전한 교체: vector_store.tmp 에 먼저 만들고 건수까지 검증한 뒤에만 기존 DB와 교체한다.
   (예전에는 빌드 전에 기존 DB를 먼저 지워서, 빌드 실패 시 DB가 사라졌다.)
 - 빌드 전에 품목표(CSV)·규정 원문 정합성 점검(check_data_consistency)을 출력한다. --strict 이면 오류 시 중단.

사용:  python src/build_vector_store.py [--strict]
"""
import gc
import json
import re
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "kc_exemption.txt"
CHROMA_DIR = ROOT_DIR / "vector_store"
TMP_DIR = ROOT_DIR / "vector_store.tmp"
OLD_DIR = ROOT_DIR / "vector_store.old"
META_NAME = "embedding_meta.json"

# 벡터 스토어 메타데이터 구조 버전. 청크 메타데이터의 키/의미를 바꾸면 올리고, rag_store 가 DB와 코드의 버전이 다르면 검색을 막는다.
#   2: scope(specific|general) + hsk6_<앞6자리> 플래그 + law_type
SCHEMA_VERSION = 2

CATEGORY = "customs_and_certification"
SOURCE_NAME = "kc_exemption.txt"
MAX_CHARS = 900

_SECTION_RE = re.compile(r"^(\d+)\.\s+(.+?)\s*$", re.MULTILINE)  # 들여쓰기 없는 '7. 제목' 만 조항 머리로 인정
_HSK_RE = re.compile(r"(?<!\d)\d{10}(?!\d)")
_COMMON_TITLE_RE = re.compile(r"절차|지침")


# ── 순수 함수 (외부 패키지 불필요, 단위 테스트 대상) ────────────────────────────
def parse_sections(text: str) -> list:
    """원문을 [{'number', 'title', 'body'}] 로 나눈다. 번호는 1부터 순서대로 이어질 때만 조항 머리로 본다."""
    headers = []
    expected = 1
    for match in _SECTION_RE.finditer(text):
        if int(match.group(1)) == expected:
            headers.append(match)
            expected += 1

    sections = []
    preface = text[: headers[0].start()].strip() if headers else text.strip()
    if preface:
        first_line = preface.splitlines()[0].strip().strip("[]")
        sections.append({"number": 0, "title": first_line, "body": preface})
    for idx, match in enumerate(headers):
        end = headers[idx + 1].start() if idx + 1 < len(headers) else len(text)
        sections.append({
            "number": int(match.group(1)),
            "title": match.group(2),
            "body": text[match.end():end].strip(),
        })
    return sections


def classify_law_type(number: int, title: str, body: str) -> str:
    if number == 0 or _COMMON_TITLE_RE.search(title):
        return "common"
    haystack = f"{title}\n{body}"
    radio = "전파법" in haystack or "방송통신기자재" in haystack
    electrical = "전기용품 및 생활용품" in haystack or "전안법" in haystack
    if radio and electrical:
        return "complex"
    if radio:
        return "radio"
    if electrical:
        return "electrical"
    return "common"


def extract_hsk_codes(text: str) -> list:
    seen = []
    for code in _HSK_RE.findall(text):
        if code not in seen:
            seen.append(code)
    return seen


def _split_body(header: str, body: str, max_chars: int) -> list:
    """긴 조항은 줄 단위로 묶어 나눈다. 나뉜 청크마다 조항 머리를 붙여 문맥을 유지한다."""
    if len(header) + 1 + len(body) <= max_chars:
        return [f"{header}\n{body}".strip()]
    chunks, current = [], ""
    for line in body.splitlines():
        candidate = f"{current}\n{line}" if current else line
        if current and len(header) + 1 + len(candidate) > max_chars:
            chunks.append(f"{header}\n{current}")
            current = line
        else:
            current = candidate
    if current:
        chunks.append(f"{header}\n{current}")
    return chunks


def build_documents(text: str, max_chars: int = MAX_CHARS) -> list:
    """[(page_content, metadata)] 목록. metadata 값은 Chroma 가 허용하는 str/int/float/bool 만 쓴다."""
    documents = []
    for section in parse_sections(text):
        law_type = classify_law_type(section["number"], section["title"], section["body"])
        header = section["title"] if section["number"] == 0 else f"{section['number']}. {section['title']}"
        flags = {}
        if law_type != "common":
            for code in extract_hsk_codes(f"{section['title']}\n{section['body']}"):
                flags[f"hsk6_{code[:6]}"] = True
        # HSK 코드가 있는 조항은 'specific'(품목 전용), 없으면 'general'(법령 일반·절차·지침)
        scope = "specific" if flags else "general"

        for part, chunk in enumerate(_split_body(header, section["body"], max_chars)):
            metadata = {
                "category": CATEGORY,
                "source": SOURCE_NAME,
                "section": section["number"],
                "part": part,
                "title": header,
                "law_type": law_type,
                "scope": scope,
            }
            metadata.update(flags)
            documents.append((chunk, metadata))
    return documents


# ── 빌드 (chromadb / langchain 필요) ───────────────────────────────────────────
def _release_chroma_handles() -> None:
    try:
        from chromadb.api.shared_system_client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception:  # noqa: BLE001 - 버전에 따라 없을 수 있음
        pass
    gc.collect()


def _swap_into_place() -> bool:
    """tmp → 정식 경로. 실패하면 기존 DB를 되돌리고 tmp 는 남긴다."""
    if OLD_DIR.exists():
        shutil.rmtree(OLD_DIR, ignore_errors=True)
    try:
        if CHROMA_DIR.exists():
            CHROMA_DIR.rename(OLD_DIR)
    except OSError as exc:
        print(f"[오류] 기존 벡터 스토어를 교체하지 못했습니다 ({exc}). Streamlit/Python 프로세스를 종료한 뒤 다시 실행하세요.")
        print(f"       새로 만든 DB는 {TMP_DIR} 에 남아 있고, 기존 DB는 그대로입니다.")
        return False
    try:
        TMP_DIR.rename(CHROMA_DIR)
    except OSError as exc:
        print(f"[오류] 새 벡터 스토어 배치 실패 ({exc}). 기존 DB를 복구합니다.")
        if OLD_DIR.exists():
            OLD_DIR.rename(CHROMA_DIR)
        return False
    shutil.rmtree(OLD_DIR, ignore_errors=True)
    return True


def _print_consistency(strict: bool) -> bool:
    """정합성 점검 결과를 출력. strict 이고 오류가 있으면 False."""
    try:
        from check_data_consistency import run_checks
    except Exception as exc:  # noqa: BLE001
        print(f"[알림] 정합성 점검을 건너뜁니다 ({exc})")
        return True
    findings = run_checks()
    errors = [f for f in findings if f[0] == "ERROR"]
    for level, message in findings:
        print(f"   [{level}] {message}")
    if errors and strict:
        print("[중단] --strict: 데이터 불일치를 먼저 해결한 뒤 다시 실행하세요.")
        return False
    return True


def build_vector_database(strict: bool = False) -> bool:
    if not DATA_PATH.exists():
        print(f"[오류] 규정 원문 파일이 존재하지 않습니다: {DATA_PATH}")
        return False

    print("0. 데이터 정합성 점검 (품목표 CSV ↔ 규정 원문)")
    if not _print_consistency(strict):
        return False

    from langchain_core.documents import Document

    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma

    from embeddings import embedding_signature, get_embeddings

    print("1. 규정 원문 로드 및 조항 단위 분할 중...")
    text = DATA_PATH.read_text(encoding="utf-8")
    built = build_documents(text)
    documents = [Document(page_content=content, metadata=meta) for content, meta in built]
    for _, meta in built:
        flags = sorted(k for k in meta if k.startswith("hsk6_"))
        print(f"   - §{meta['section']}.{meta['part']} {meta['law_type']:<10} {flags} {meta['title'][:40]}")
    print(f"   총 {len(documents)}개 청크")

    print("2. 임베딩 + 임시 폴더에 벡터 스토어 생성 중 (기존 DB는 건드리지 않음)...")
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR, ignore_errors=True)
    TMP_DIR.mkdir(parents=True)
    store = Chroma.from_documents(
        documents=documents,
        embedding=get_embeddings(),
        persist_directory=str(TMP_DIR),
    )
    stored = store._collection.count()  # noqa: SLF001
    if stored != len(documents):
        print(f"[오류] 저장 건수 불일치 (기대 {len(documents)} / 실제 {stored}). 기존 DB를 유지합니다.")
        return False
    (TMP_DIR / META_NAME).write_text(
        json.dumps({**embedding_signature(), "schema": SCHEMA_VERSION}, ensure_ascii=False), encoding="utf-8"
    )
    del store
    _release_chroma_handles()

    print("3. 기존 DB와 교체 중...")
    if not _swap_into_place():
        return False
    print(f"[성공] 벡터 스토어를 재구축했습니다: {CHROMA_DIR}")
    return True


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ok = build_vector_database(strict="--strict" in sys.argv)
    sys.exit(0 if ok else 1)
