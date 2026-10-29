"""Streamlit 화면 (app.py 수정본)

이번 수정
 1. 시연 결과: 예전에는 질문이 "연구개발 목적으로 리튬이온"으로 '시작하기만' 하면 분석 없이 고정 캐시를 보여 줬다.
    이제 기본 질문과 '완전히 같을 때만' demo/demo_default.json 의 저장 결과를 쓰고, 화면에 "저장된 시연 결과(실시간 분석 아님)"라고 밝힌다.
    사이드바의 체크를 풀면 기본 질문도 실시간으로 분석한다.
 2. 캐시 초기화: 예전 코드는 ROOT_DIR.parent / "cache" (프로젝트의 상위 폴더)를 지워 아무것도 지우지 못했다.
    실제 캐시 폴더(프로젝트/cache)를 지우고, 지운 건수를 보여 준다. 중복 버튼은 하나로 정리했다.
 3. 모바일 입력: 입력창과 실행 버튼을 st.form 으로 묶어, 입력 직후 버튼을 눌러도 최신 입력으로 실행된다.
 4. 한도 초과·서버 혼잡은 경고(노란색)로, 그 밖의 실패는 에러(빨간색)로 구분해 보여 준다.
 5. 3.4MB 아이콘을 그대로 보내지 않고 작은 썸네일로 줄여서 쓴다 (모바일 로딩 속도).
 6. 주소 끝에 ?debug=1 을 붙이면 진단 영역(모델, 키 끝자리, API 1회 테스트)이 보인다. 오류 원문의 API 키는 지운다.
"""
import functools
import json
import os
import re
import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "src"))

import config  # noqa: E402

CACHE_DIR = ROOT_DIR / "cache"
DEMO_FILE = ROOT_DIR / "demo" / "demo_default.json"
ICON_FILE = ROOT_DIR / "AI_Agent_icon.png"

DEFAULT_QUERY = (
    "연구개발 목적으로 리튬이온 배터리 프로토타입 100개를 수입하려고 해. "
    "개당 가격은 5천원이고 인증은 아직 없어. 전안법 면제 조건과 관세율 알려줘."
)
STAGE_LABELS = {
    "input": "입력 확인",
    "rule": "사전 규칙 검사",
    "router": "의도 분류",
    "hsk": "HSK 추론",
    "report": "규칙 검사·보고서 생성",
}
TRANSIENT_KINDS = ("quota", "overloaded")
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


# ── 도우미 ──────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=1)
def _load_icon():
    """3.4MB 원본 대신 160px 썸네일을 만든다. 실패하면 None (이모지로 대체)."""
    if not ICON_FILE.exists():
        return None
    try:
        from PIL import Image

        with Image.open(ICON_FILE) as image:
            thumb = image.convert("RGBA")
            thumb.thumbnail((160, 160))
            return thumb.copy()
    except Exception:  # noqa: BLE001
        return None


def _norm(text) -> str:
    return " ".join(str(text or "").split())


def _load_demo(query: str):
    """기본 질문과 '완전히' 같을 때만 저장된 시연 결과를 돌려준다. 아니면 None."""
    if not DEMO_FILE.exists():
        return None
    try:
        payload = json.loads(DEMO_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or _norm(query) != _norm(payload.get("query")):
        return None
    result = payload.get("result")
    if not isinstance(result, dict) or result.get("status") != "success":
        return None
    demo = dict(result)
    demo["demo"] = True
    demo["demo_generated_at"] = payload.get("generated_at", "")
    return demo


def _clear_cache() -> int:
    """실제 캐시 폴더(프로젝트/cache)의 저장 결과를 지우고 삭제한 개수를 돌려준다."""
    removed = 0
    if CACHE_DIR.exists():
        for path in list(CACHE_DIR.glob("cache_*.json")) + list(CACHE_DIR.glob("*.tmp")):
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
    return removed


def _redact(text: str) -> str:
    out = str(text)
    key = os.getenv("GEMINI_API_KEY", "")
    if len(key) >= 8:
        out = out.replace(key, "***")
    return re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "***", out)


def _debug_enabled() -> bool:
    params = getattr(st, "query_params", None)
    try:
        return params is not None and params.get("debug") == "1"
    except Exception:  # noqa: BLE001
        return False


# ── 페이지 설정 (첫 Streamlit 명령) ──────────────────────────────────────────
ICON = _load_icon()
st.set_page_config(
    page_title="R&D 패스트트랙 AI AGENT",
    page_icon=ICON if ICON is not None else "🚀",
    layout="wide",
)

title_col1, title_col2 = st.columns([1.5, 10])
with title_col1:
    if ICON is not None:
        st.image(ICON, width=70)
with title_col2:
    st.header("R&D 패스트트랙 AI AGENT")

st.markdown(
    "소규모 하드웨어 스타트업 및 연구개발(R&D) 목적의 시제품 수입을 위한 "
    "**HSK 추론, 관세율 조회, KC 인증 면제 사유서 자동 생성** 솔루션입니다."
)

st.sidebar.header("📌 데모 안내")
st.sidebar.markdown("""
- **처리 단계**:
  1. 의도 분류 (수입 관련 표현 + 수량이면 규칙, 그 밖은 AI)
  2. 사전 규칙 검사 (상업용, 수량·단가 오류 즉시 차단)
  3. HSK 코드 추론 (로컬 품목표 후보군 중 선택)
  4. 수량 한도 검증 + 세액 산출
  5. 메타데이터 RAG 기반 규정 참조 + 사유서 작성
  6. 워드(`.docx`) 및 마크다운(`.md`) 자동 저장
""")

use_template = st.sidebar.checkbox(
    "사유서 템플릿 모드 적용\n(체크 시 사유서 작성에 API를 쓰지 않아 호출 1회로 최적화)",
    value=True,
)
use_demo = st.sidebar.checkbox(
    "기본 질문은 저장된 시연 결과 사용\n(API 호출 없음, 실시간 분석 아님)",
    value=True,
)

# ── 입력부: 입력창과 버튼을 form 으로 묶는다 (모바일에서 입력 직후 버튼을 눌러도 최신 값으로 실행) ──
with st.form("query_form", clear_on_submit=False):
    query = st.text_area(
        "수입하고자 하는 품목과 연구개발 목적, 수량, 단가를 자유롭게 입력해 주세요:",
        value=DEFAULT_QUERY,
        height=200,
    )
    run_clicked = st.form_submit_button("🔍 통관 및 인증 분석 실행", type="primary", use_container_width=True)

if st.button("🧹 캐시 초기화", use_container_width=True):
    removed = _clear_cache()
    st.session_state.pop("result", None)
    st.success(f"저장된 분석 캐시 {removed}건을 삭제했습니다.")

# ── 실행 ────────────────────────────────────────────────────────────────────
if run_clicked:
    if not query.strip():
        st.warning("질문을 입력해 주세요.")
    else:
        demo_result = _load_demo(query) if use_demo else None
        if demo_result is not None:
            st.session_state["result"] = demo_result
        else:
            with st.spinner("AI 에이전트가 질문을 분석하고 있습니다..."):
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
    """마크다운은 결과 문자열에서 바로 내려받게 한다(서버에 파일이 없어도 동작). 워드는 서버에 파일이 있을 때만."""
    cols = st.columns(2)
    md_path = result.get("md_path")
    if md_path and Path(md_path).exists():
        md_data, md_name = Path(md_path).read_bytes(), Path(md_path).name
    else:
        md_data, md_name = result.get("report_markdown", "").encode("utf-8"), "tariff_report.md"
    cols[0].download_button("📄 마크다운 다운로드", data=md_data, file_name=md_name, mime="text/markdown", key="download_md")

    docx_path = result.get("docx_path")
    if docx_path and Path(docx_path).exists():
        cols[1].download_button(
            "📝 워드 다운로드", data=Path(docx_path).read_bytes(), file_name=Path(docx_path).name,
            mime=DOCX_MIME, key="download_docx",
        )


# ── 결과 출력부 ──────────────────────────────────────────────────────────
result = st.session_state.get("result")
if result:
    status = result.get("status")
    if status == "success":
        if result.get("demo"):
            generated = result.get("demo_generated_at") or "일시 미상"
            st.info(
                f"📌 **저장된 시연 결과입니다 (실시간 분석이 아닙니다).** 생성: {generated}. "
                "실시간 분석을 하려면 질문을 바꾸거나 사이드바의 '저장된 시연 결과 사용'을 해제하세요."
            )
        else:
            st.success("✅ 분석 및 문서 생성이 완료되었습니다!")
            if result.get("cache_hit"):
                st.caption("같은 질문의 이전 분석 결과를 불러왔습니다. 새로 분석하려면 '캐시 초기화'를 누르세요.")
        _show_warnings(result)
        _show_meta(result)
        st.markdown("---")
        
        # 👇 추가된 이미지 출력 코드
        report_icon = ROOT_DIR / "AI_Agent_보고서.png"
        if report_icon.exists():
            st.image(str(report_icon), width=80)
            
        st.markdown(result["report_markdown"])
        if result.get("docx_path") and Path(result["docx_path"]).exists():
            st.info(f"📂 생성된 파일은 서버의 `output/` 폴더에 저장되었습니다. (경로: {result['docx_path']})")
        _download_buttons(result)
    else:
        message = result.get("message", "알 수 없는 오류")
        if result.get("error_kind") in TRANSIENT_KINDS:
            st.warning(f"⏳ {message}")
        else:
            stage = STAGE_LABELS.get(result.get("stage"), "알 수 없음")
            st.error(f"분석 실패 ({stage} 단계): {message}")
        _show_meta(result)

# ── 진단 영역 (?debug=1 일 때만) ─────────────────────────────────────────────
if _debug_enabled():
    with st.sidebar.expander("🛠 진단 (개발자용)", expanded=True):
        key = os.getenv("GEMINI_API_KEY", "")
        key_hint = f"…{key[-4:]}" if len(key) >= 8 else "없음 또는 너무 짧음"
        st.caption(f"모델 `{getattr(config, 'MODEL_NAME', '?')}` · API 키 끝자리 `{key_hint}`")
        cached = len(list(CACHE_DIR.glob("cache_*.json"))) if CACHE_DIR.exists() else 0
        st.caption(f"저장된 분석 캐시 {cached}건 · 시연 파일 {'있음' if DEMO_FILE.exists() else '없음'}")
        if st.button("API 1회 테스트", key="api_test"):
            try:
                reply = config.generate_text("OK 라고만 답하세요.")
                st.success(f"정상 응답: {reply[:40]}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"{type(exc).__name__}: {_redact(str(exc))[:400]}")
