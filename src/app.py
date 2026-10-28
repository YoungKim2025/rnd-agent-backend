"""Streamlit 화면 (모바일 최적화 및 중복 버튼 제거 최종본)"""
import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "src"))

STAGE_LABELS = {
    "input": "입력 확인",
    "rule": "사전 규칙 검사",
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
  1. 사전 규칙 검사 (상업용, 단가 오류 즉시 차단)
  2. HSK 코드 추론 (로컬 품목표 후보군 중 선택)
  3. 수량 한도 검증 + 세액 산출
  4. 메타데이터 RAG 기반 규정 참조 + 사유서 작성
  5. 워드(`.docx`) 및 마크다운(`.md`) 자동 저장
""")

use_template = st.sidebar.checkbox(
    "사유서 템플릿 모드 적용\n(체크 시 API 호출 1회로 최적화)", 
    value=True
)

if st.sidebar.button("🧹 에러 캐시 및 세션 초기화"):
    import shutil
    cache_dir = ROOT_DIR.parent / "cache"
    if cache_dir.exists():
        for f in cache_dir.glob("*.json"):
            try:
                f.unlink()
            except Exception:
                pass
    st.session_state.clear()
    st.success("캐시와 세션이 초기화되었습니다. 다시 시도해 주세요.")
    st.rerun()

# ── 메인 화면 입력부 ──────────────────────────────────────────────────────
query = st.text_area(
    "수입하고자 하는 품목과 연구개발 목적, 수량, 단가를 자유롭게 입력해 주세요:",
    value="연구개발 목적으로 리튬이온 배터리 프로토타입 100개를 수입하려고 해. 개당 가격은 5천원이고 인증은 아직 없어. 전안법 면제 조건과 관세율 알려줘.",
    height=200,
)

# ── 메인 실행 및 초기화 버튼 (나란히 배치) ──────────────────────────────────
col_btn1, col_btn2 = st.columns([3, 1])
with col_btn1:
    run_clicked = st.button("🔍 통관 및 인증 분석 실행", type="primary", use_container_width=True)
with col_btn2:
    if st.button("🧹 캐시 초기화", use_container_width=True):
        import shutil
        cache_dir = ROOT_DIR.parent / "cache"
        if cache_dir.exists():
            for f in cache_dir.glob("*.json"):
                try:
                    f.unlink()
                except Exception:
                    pass
        st.session_state.clear()
        st.success("캐시와 세션이 초기화되었습니다!")
        st.rerun()

# ── 실행 로직 (중복 제거 완료) ───────────────────────────────────────────
if run_clicked:
    if not query.strip():
        st.warning("질문을 입력해 주세요.")
    else:
        with st.spinner("AI 에이전트가 캐시 및 규정을 검토하여 분석을 진행하고 있습니다..."):
            from pipeline_runner import run_pipeline
            st.session_state["result"] = run_pipeline(query, use_template=use_template)


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


# ── 결과 출력부 ──────────────────────────────────────────────────────────
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