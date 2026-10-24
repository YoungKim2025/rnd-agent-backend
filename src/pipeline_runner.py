"""R&D 패스트트랙 통합 실행기 (권장 순서 1번)

의도 분류(IntentRouter) -> HSK 추론(infer_hsk_code) -> 보고서 생성(generate_fast_track_report)
3단계를 run_pipeline(query) 하나로 묶어, app.py / test_run.py가 같은 진입점을 쓰도록 한다.

* 기존 main_pipeline.py / intent_router.py / hsk_inferencer.py 는 수정하지 않고 import만 한다.
* 이 파일을 src/ 폴더에 pipeline_runner.py 이름으로 추가해서 사용한다.

반환값(dict) 공통 규칙
    status  : "success" | "error" | "unsupported"
    stage   : 끝난 단계 - "input" | "router" | "hsk" | "report" | "done"
    message : status가 success가 아닐 때 사용자에게 보여줄 문장
    route / inference : 의도 분류, HSK 추론 정보 (해당 단계까지 진행됐을 때만 포함)
    status가 success이면 generate_fast_track_report()의 반환값
    (report_markdown, md_path, docx_path)이 그대로 함께 들어 있다.

단독 실행:
    python src/pipeline_runner.py "연구개발 목적으로 리튬이온 배터리 25대 수입하려고 해"
"""
import logging
import sys
from pathlib import Path

# main_pipeline.py 가 'from intent_router import ...' 형태로 import 하므로 src 폴더를 경로에 보장한다.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from intent_router import IntentRouter, CUSTOMS, LAB_SAFETY  # noqa: E402
from hsk_inferencer import infer_hsk_code  # noqa: E402
from main_pipeline import generate_fast_track_report  # noqa: E402

logger = logging.getLogger("rnd_pipeline")

_router = None  # IntentRouter는 Client를 만들기 때문에 처음 한 번만 생성한다.


def _get_router():
    global _router
    if _router is None:
        _router = IntentRouter()  # GEMINI_API_KEY가 없으면 RuntimeError
    return _router


def _fail(stage: str, message: str, status: str = "error", **extra) -> dict:
    result = {"status": status, "stage": stage, "message": message}
    result.update(extra)
    return result


def run_pipeline(query: str) -> dict:
    """사용자 질문 한 건을 처리해 보고서(md/docx)까지 생성한다. 예외는 밖으로 던지지 않는다."""
    text = str(query or "").strip()
    if not text:
        return _fail("input", "질문을 입력해 주세요.")

    # ── 1단계: 의도 분류 ────────────────────────────────────────────
    try:
        router = _get_router()
    except RuntimeError as e:  # 키 미설정
        return _fail("router", str(e))

    try:
        route = router.route(text)
    except Exception as e:
        logger.exception("의도 분류 실패")
        return _fail("router", f"의도 분류 중 오류가 발생했습니다 ({type(e).__name__}).")

    route_info = {
        "category": route.category,
        "confidence": route.confidence,
        "reason": route.reason,
        "fallback_used": route.fallback_used,
    }

    if route.category == LAB_SAFETY:
        return _fail(
            "router",
            "연구실 안전 문의는 아직 지원하지 않습니다(Phase 2 예정). 현재는 수입 통관·인증 질문만 처리합니다.",
            status="unsupported", route=route_info,
        )
    if route.category != CUSTOMS:
        return _fail(
            "router",
            "수입 통관·인증과 관련된 질문이 아닙니다. 수입 품목, 수량, 연구개발 목적을 포함해 다시 입력해 주세요.",
            status="unsupported", route=route_info,
        )

    # ── 2단계: HSK 추론 ─────────────────────────────────────────────
    infer = infer_hsk_code(text)
    if not isinstance(infer, dict) or infer.get("status") != "success":
        message = infer.get("message") if isinstance(infer, dict) else None
        return _fail("hsk", message or "HSK 추론에 실패했습니다.", route=route_info)

    data = infer.get("data")
    if not isinstance(data, dict):
        return _fail("hsk", "HSK 추론 결과 형식이 올바르지 않습니다.", route=route_info)

    hsk_code = str(data.get("hsk_code") or "").strip()
    inference_info = {
        "hsk_code": hsk_code,
        "item_name": data.get("item_name"),
        "confidence": data.get("confidence"),
        "reason": data.get("reason"),
    }
    if not hsk_code:
        reason = data.get("reason") or "근거 없음"
        return _fail(
            "hsk",
            f"지원 대상 품목을 특정하지 못했습니다. 품목명을 구체적으로 적어 주세요. (사유: {reason})",
            route=route_info, inference=inference_info,
        )

    # ── 3단계: 관세율 조회 + RAG + 사유서 + 파일 저장 ───────────────────
    try:
        report = generate_fast_track_report(text, hsk_hint=hsk_code)
    except Exception as e:
        logger.exception("보고서 생성 실패")
        return _fail(
            "report", f"보고서 생성 중 오류가 발생했습니다 ({type(e).__name__}).",
            route=route_info, inference=inference_info,
        )

    if not isinstance(report, dict) or report.get("status") != "success":
        message = report.get("message") if isinstance(report, dict) else None
        return _fail(
            "report", message or "보고서를 생성하지 못했습니다.",
            route=route_info, inference=inference_info,
        )

    result = dict(report)
    result.update({"stage": "done", "route": route_info, "inference": inference_info})
    return result


if __name__ == "__main__":
    sample = "연구개발 목적으로 리튬이온 배터리 프로토타입 25대를 수입하려고 해."
    res = run_pipeline(" ".join(sys.argv[1:]) or sample)
    print(f"status={res['status']} stage={res['stage']}")
    if res["status"] == "success":
        print("MD   :", res["md_path"])
        print("DOCX :", res["docx_path"])
    else:
        print(res["message"])
