"""관세청 API MCP 서버 (mcp_server.py 대체본)

변경점
 - customs_api.py 가 이제 실제 API 결과(또는 status=error)만 돌려주므로 가짜 세율이 MCP 로 노출되지 않는다.
 - MCP SDK 버전에 따라 import 경로가 달라서 두 경로를 모두 시도한다.
     mcp.server.mcpserver.MCPServer  (2.x 표기)  /  mcp.server.fastmcp.FastMCP (1.x)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

_Server = None
try:
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:
    try:
        from mcp.server.fastmcp import FastMCP as _Server
    except ImportError as exc:
        print(f"모듈 로드 에러: {exc} (pip install mcp)", file=sys.stderr)
        sys.exit(1)

from customs_api import fetch_tariff_from_customs_api  # noqa: E402

mcp = _Server("Customs Agent")


@mcp.tool()
def get_customs_tariff(hsk_code: str) -> str:
    """관세청 API를 호출하여 관세율(기본, WTO, FTA 등) 조회. 실패 시 status=error 와 사유를 반환한다."""
    try:
        result = fetch_tariff_from_customs_api(hsk_code)
    except Exception as exc:  # noqa: BLE001
        result = {"status": "error", "message": str(exc)}
    return json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
