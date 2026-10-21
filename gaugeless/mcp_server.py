"""MCP server: the Gaugeless measurement tools for any MCP client (Claude Desktop, Claude Code, Bedrock agents).

    python -m gaugeless.mcp_server                 # stdio
    python -m gaugeless.mcp_server --http 8765     # streamable HTTP

Same tool functions and guard rails as the in-process planners (gaugeless/agent/tools.py). There is deliberately
no `approve` tool: human approval comes from the operator channel (Step Functions task token / web UI),
so an agent can never approve its own result.
"""
from __future__ import annotations

import argparse
import uuid

from mcp.server.mcpserver import MCPServer

from .agent import tools as T
from .agent.llm import _clean
from .agent.run import make_session
from .cases import CASES

mcp = MCPServer("gaugeless", instructions=(
    "Measure river surface velocity and discharge from video. Start with open_session(site), then load_clip and "
    "diagnose_clip. Never publish a number the checks reject; refuse or escalate instead."))
SESSIONS: dict[str, T.Session] = {}


def _s(session_id: str) -> T.Session:
    if session_id not in SESSIONS:
        raise ValueError(f"unknown session {session_id}; call open_session first")
    return SESSIONS[session_id]


@mcp.tool()
def list_sites() -> list[dict]:
    """Sites with video on disk: name, platform, frame rate, ground sampling distance, gauge points, sections."""
    out = []
    for n, c in CASES.items():
        if n == "alpine_full" or c.available() == 0:
            continue
        out.append({"site": n, "platform": c.platform, "fps": c.fps, "gsd_m_per_px": c.gsd, "frames": c.available(),
                    "has_sections": n == "lamorge"})
    return out


@mcp.tool()
def open_session(site: str, flood_threshold_m3s: float | None = None) -> dict:
    """Start a measurement session for a site. Returns session_id used by every other tool."""
    sid = uuid.uuid4().hex[:12]
    s = make_session(site, flood_threshold_m3s)
    SESSIONS[sid] = s
    return {"session_id": sid, "site": site, "gauge_points": 0 if s.targets is None else len(s.targets),
            "sections": list(s.sections or [])}


def _register(name):
    fn = T.TOOLS[name]
    spec = next(t for t in T.TOOL_SPECS if t["name"] == name)

    def tool(session_id: str, args: dict | None = None) -> dict:
        return _clean(fn(_s(session_id), **(args or {})))
    tool.__name__ = name
    props = spec["input_schema"].get("properties", {})
    arg_doc = ", ".join(f"{k} ({v.get('type')}{', one of ' + str(v['enum']) if 'enum' in v else ''})"
                        for k, v in props.items())
    mcp.tool(name=name, description=spec["description"] + (f"\nargs: {arg_doc}" if arg_doc else ""))(tool)


for _n in T.TOOLS:
    _register(_n)


@mcp.tool()
def get_trace(session_id: str) -> list[dict]:
    """Every tool call in this session with its arguments, result and duration."""
    return _clean(_s(session_id).trace)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", type=int)
    a = ap.parse_args()
    if a.http:
        mcp.run("streamable-http", port=a.http)
    else:
        mcp.run("stdio")
