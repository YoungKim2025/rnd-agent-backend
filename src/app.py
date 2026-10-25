"""Streamlit 화면 (app.py 대체본)

변경점
 - 보고서의 '확인이 필요한 사항'(단가 가정, 수량 한도 미등록, 목적 미확인 등)을 화면 상단에 경고로 표시
 - 사이드바 설명을 실제 동작(기본세율 기준, 원산지증명서 입력 시 FTA 반영, 단가 입력)에 맞게 수정
 - 그 외 구조(run_pipeline 호출, session_state 보관, 단계별 실패 표시)는 이전과 동일
"""
import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "src"))

STAGE_LABELS = {
    "input": "입력 확인",
    "router": "의도 분류",
    "hsk": "HSK 추론",
    "report": "보고서 생성",
}
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

st.set_page_config(
    page_title="R&D 패스트트랙 AI AGENT",
    page_icon="🚀",
    layout="wide",
)

st.header("🚀 R&D 패스트트랙 AI AGENT")
st.markdown(
    "소규모 하드웨어 스타트업 및 연구개발(R&D) 목적의 시제품 수입을 위한 "
    "**HSK 추론, 관세율 조회, KC 인증 면제 사유서 자동 생성** 솔루션입니다."
)

st.sidebar.header("📌 데모 안내")
st.sidebar.markdown("""
- **처리 단계**:
  1. 질문 의도 분류 (통관·인증 / 연구실 안전 / 기타)
  2. HSK 코드 추론 (로컬 품목표 후보군 중 선택)
  3. 수량 한도 검증 + 세액 산출 (기본세율 기준, 원산지증명서를 입력하면 FTA 세율 반영)
  4. ChromaDB RAG 기반 규정 참조 + 사유서 작성
  5. 워드(`.docx`) 및 마크다운(`.md`) 자동 저장
- **입력 팁**: 품목, 수량(`25개`), 단가(`개당 5천원`), 수입 목적(연구개발 등)을 함께 적어 주세요.
  단가를 쓰지 않으면 가정값으로 계산하고 보고서에 표시합니다.
- **지원 품목**: `data/tariff_rates.csv`에 등록된 품목
""")

query = st.text_area(
    "수입하고자 하는 품목과 연구개발 목적, 수량, 단가를 자유롭게 입력해 주세요:",
    value="연구개발 목적으로 리튬이온 배터리 프로토타입 100개를 수입하려고 해. 개당 가격은 5천원이고 인증은 아직 없어. 전안법 면제 조건과 관세율 알려줘.",
    height=200,
)


def _show_meta(result: dict) -> None:
    route = result.get("route") or {}
    inference = result.get("inference") or {}
    parts = []
    if route:
        parts.append(f"의도 분류: {route.get('category')} (신뢰도 {route.get('confidence')})")
    if inference.get("hsk_code"):
        parts.append(
            f"HSK 추론: {inference['hsk_code']} {inference.get('item_name') or ''} "
            f"(신뢰도 {inference.get('confidence')})"
        )
    if parts:
        st.caption(" | ".join(parts))


def _show_warnings(result: dict) -> None:
    for warning in result.get("warnings") or []:
        st.warning(warning)


def _download_buttons(result: dict) -> None:
    targets = (
        ("md_path", "📄 마크다운 다운로드", "text/markdown"),
        ("docx_path", "📝 워드 다운로드", DOCX_MIME),
    )
    cols = st.columns(len(targets))
    for col, (key, label, mime) in zip(cols, targets):
        path = result.get(key)
        if path and Path(path).exists():
            col.download_button(
                label,
                data=Path(path).read_bytes(),
                file_name=Path(path).name,
                mime=mime,
                key=f"download_{key}",
            )


if st.button("🔍 통관 및 인증 분석 실행", type="primary"):
    if not query.strip():
        st.warning("질문을 입력해 주세요.")
    else:
        # 최초 실행 시 모듈 로드 시간을 감안한 스피너 문구로 변경
        with st.spinner("AI 엔진을 초기화하고 분석을 진행하고 있습니다. (최초 1회 수 초 소요)..."):
            from pipeline_runner import run_pipeline  # 지연 로딩 (Lazy Loading)
            st.session_state["result"] = run_pipeline(query)

result = st.session_state.get("result")
if result:
    status = result.get("status")
    if status == "success":
        st.success("✅ 분석 및 문서 생성이 완료되었습니다!")
        _show_warnings(result)
        _show_meta(result)
        st.markdown("---")
        st.markdown(result["report_markdown"])
        st.info(f"📂 생성된 파일은 `output/` 폴더에도 저장되었습니다. (경로: {result.get('docx_path')})")
        _download_buttons(result)
    elif status == "unsupported":
        st.info(result.get("message", "지원하지 않는 질문입니다."))
        _show_meta(result)
    else:
        stage = STAGE_LABELS.get(result.get("stage"), "알 수 없음")
        st.error(f"분석 실패 ({stage} 단계): {result.get('message', '알 수 없는 오류')}")
        _show_meta(result)
