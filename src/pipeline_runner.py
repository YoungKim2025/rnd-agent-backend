"""R&D 패스트트랙 통합 실행기 (에러 캐시 원천 차단 및 불량 캐시 자동 삭제 적용)"""
import hashlib
import json
import logging
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import config
import report_logic as rl
from intent_router import IntentRouter, CUSTOMS, LAB_SAFETY
from hsk_inferencer import infer_hsk_code
from main_pipeline import generate_fast_track_report, CSV_PATH

logger = logging.getLogger("rnd_pipeline")
_router = None

CACHE_DIR = _SRC_DIR.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_router():
    global _router
    if _router is None:
        _router = IntentRouter()
    return _router


def _fail(stage: str, message: str, status: str = "error", **extra) -> dict:
    result = {"status": status, "stage": stage, "message": message}
    result.update(extra)
    return result


def _save_cache(path: Path, data: dict):
    """성공한 결과(status == 'success')일 때만 최종적으로 캐시를 저장"""
    if data.get("status") != "success":
        return
        
    try:
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.warning(f"캐시 저장 실패: {e}")


def _get_csv_hash() -> str:
    if CSV_PATH.exists():
        return hashlib.md5(CSV_PATH.read_bytes()).hexdigest()
    return "no_csv"


def _get_logic_hash() -> str:
    targets = [
        _SRC_DIR / "main_pipeline.py",
        _SRC_DIR / "report_logic.py",
        _SRC_DIR / "hsk_inferencer.py",
        _SRC_DIR / "intent_router.py",
        _SRC_DIR.parent / "data" / "kc_exemption.txt",
    ]
    h = hashlib.md5()
    for t in targets:
        if t.exists():
            h.update(t.read_bytes())
        else:
            h.update(b"missing")
    return h.hexdigest()


def run_pipeline(query: str, use_template: bool = True) -> dict:
    text = str(query or "").strip()
    if not text:
        return _fail("input", "질문을 입력해 주세요.")

    model_name = getattr(config, "MODEL_NAME", "gemini-3.8-flash")
    logic_hash = _get_logic_hash()
    csv_hash = _get_csv_hash()
    
    # 기존 코드
    cache_raw = f"{text}|{model_name}|{logic_hash}|{csv_hash}|tmpl={use_template}"
    query_hash = hashlib.md5(cache_raw.encode('utf-8')).hexdigest()
    cache_file = CACHE_DIR / f"cache_{query_hash}.json"

    # 💡 디버깅용 출력 코드 (여기에 추가)
    print("\n" + "="*50)
    print(f"[디버그] PC에 저장된 원본 해시: e0f3814a121ab458421f3f5e4d6384ea")
    print(f"[디버그] 모바일이 요청한 해시: {query_hash}")
    print(f"[디버그] 해시 생성 원본 문자열: {cache_raw}")
    print("="*50 + "\n")

    if cache_file.exists():
        try:
            cached_data = json.loads(cache_file.read_text(encoding="utf-8"))
            
            if cached_data.get("status") == "success":
                md_path = Path(cached_data.get("md_path", ""))
                docx_path = Path(cached_data.get("docx_path", ""))
                report_md = cached_data.get("report_markdown", "")
                
                if md_path.name and not md_path.exists():
                    md_path.parent.mkdir(parents=True, exist_ok=True)
                    md_path.write_text(report_md, encoding="utf-8")
                    
                if docx_path.name and not docx_path.exists():
                    try:
                        from docx import Document
                        doc = Document()
                        doc.add_heading("R&D 패스트트랙 통관 및 세액 산출 보고서 (자동 복구됨)", level=1)
                        for line in report_md.split("\n"):
                            doc.add_paragraph(line)
                        doc.save(docx_path)
                    except Exception as e:
                        logger.warning(f"DOCX 복구 실패: {e}")
                
                logger.info("캐시된 성공 응답을 반환합니다.")
                return cached_data
            else:
                # 💡 실패한 옛날 캐시 파일이 발견되면 즉시 삭제 처리
                logger.info("에러가 기록된 기존 캐시를 발견하여 완전히 삭제 후 새로 시도합니다.")
                if cache_file.exists():
                    cache_file.unlink()
        except Exception as e:
            logger.warning(f"캐시 읽기 실패로 불량 파일을 삭제합니다: {e}")
            if cache_file.exists():
                cache_file.unlink()

    # ── 1. 규칙 검사를 최상단으로 이동 (거절 사유 즉시 차단, 불필요한 캐시 저장 없음) ──
    purpose = rl.detect_purpose(text)
    if purpose == "commercial":
        return _fail("input", "상업적 판매 목적의 수입은 R&D 패스트트랙 지원 대상이 아닙니다.")

    qty = rl.extract_quantity(text)
    if qty is None:
        return _fail("input", "수입 수량을 파악할 수 없습니다. 정확한 수량을 포함해 다시 질문해 주세요.")

    price_info = rl.extract_unit_price(text, qty)
    if price_info.get("status") == "foreign":
        return _fail("input", "외화 단가는 지원하지 않습니다. 원화(KRW) 기준으로 입력해 주세요.")

    # ── 2. 의도 분류 규칙 우선 판단 ──
    if purpose == "research" and qty is not None:
        route_info = {
            "category": CUSTOMS,
            "confidence": 1.0,
            "reason": "규칙 기반 식별(수량 및 연구 목적 확인됨)",
            "fallback_used": False
        }
    else:
        try:
            router = _get_router()
        except RuntimeError as e:
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

    if route_info["category"] == LAB_SAFETY:
        return _fail("router", "연구실 안전 문의는 아직 지원하지 않습니다(Phase 2 예정).", status="unsupported", route=route_info)
    if route_info["category"] != CUSTOMS:
        return _fail("router", "수입 통관·인증과 관련된 질문이 아닙니다.", status="unsupported", route=route_info)

    # ── 3. HSK 추론 ──
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
        return _fail("hsk", f"지원 대상 품목을 특정하지 못했습니다. (사유: {reason})", route=route_info, inference=inference_info)

    # ── 4. 사유서 템플릿 모드 및 보고서 생성 ──
    try:
        if use_template:
            original_generate = config.generate_text
            config.generate_text = lambda prompt: "본 물품은 연구개발용 시제품으로, 상업적 유통을 금지하며 사후 추적 관리를 성실히 이행할 것을 서약합니다."
            report = generate_fast_track_report(text, hsk_hint=hsk_code)
            config.generate_text = original_generate
        else:
            report = generate_fast_track_report(text, hsk_hint=hsk_code)
    except Exception as e:
        logger.exception("보고서 생성 실패")
        return _fail("report", f"보고서 생성 중 오류가 발생했습니다 ({type(e).__name__}).", route=route_info, inference=inference_info)

    if not isinstance(report, dict) or report.get("status") != "success":
        message = report.get("message") if isinstance(report, dict) else None
        return _fail("report", message or "보고서를 생성하지 못했습니다.", route=route_info, inference=inference_info)

    result = dict(report)
    result.update({"stage": "done", "route": route_info, "inference": inference_info})
    
    # ── 정상 결과 최종 캐싱 (여기서 단 한 번만 수행) ──
    _save_cache(cache_file, result)

    return result


if __name__ == "__main__":
    sample = "연구개발 목적으로 리튬이온 배터리 프로토타입 25대를 수입하려고 해."
    res = run_pipeline(" ".join(sys.argv[1:]) or sample)
    print(f"status={res['status']} stage={res.get('stage', '')}")
    if res["status"] == "success":
        print("MD   :", res.get("md_path"))
        print("DOCX :", res.get("docx_path"))
    else:
        print(res["message"])