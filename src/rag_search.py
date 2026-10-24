"""규정 검색 점검용 CLI (rag_search.py 대체본)

main_pipeline 이 실제로 쓰는 검색(rag_store.retrieve_legal_docs)을 그대로 호출해서,
어떤 조항이 검색되는지 눈으로 확인할 수 있다. (벡터 DB를 다시 만든 뒤 점검할 때 유용)

사용:
    python src/rag_search.py "연구개발 목적으로 태블릿 PC 5대 수량 한도는?" --hsk 8471300000 --law 전파법
    python src/rag_search.py "리튬이온 배터리 면제 조건" --hsk 8507600000 --law 전안법
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_store import retrieve_legal_docs  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="규정 조항 검색 점검")
    parser.add_argument("query")
    parser.add_argument("--hsk", default="", help="HSK 10자리 (품목 전용 조항 검색)")
    parser.add_argument("--law", default="", choices=["", "전파법", "전안법", "복합"], help="품목표의 law_type")
    args = parser.parse_args()

    docs = retrieve_legal_docs(args.query, args.hsk, args.law)
    if not docs:
        print("검색된 조항이 없습니다.")
        return 1
    for index, doc in enumerate(docs, 1):
        meta = doc.metadata
        print(f"[{index}] §{meta.get('section')}.{meta.get('part')} law_type={meta.get('law_type')} | {meta.get('title')}")
        print(doc.page_content)
        print("-" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
