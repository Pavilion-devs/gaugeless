"""Claude on Amazon Bedrock as the planner (Converse API with tool use). Same tools, same guard rails as rules.py.

Env: GAUGELESS_MODEL (default Claude Sonnet 4.5 inference profile), AWS_REGION (default us-east-1).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from . import tools as T

MODEL = os.environ.get("GAUGELESS_MODEL", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
MAX_TURNS = 20

SYSTEM = f"""You are the measurement agent of Gaugeless, which measures river surface velocity and discharge from video.
You decide how to measure, using only the tools. You never see reference values. Your job is a number a
hydrologist can trust, or an honest refusal or escalation. A silent wrong number is the worst outcome.

What the team measured on 14 real rivers (use it):
- Clips with moving_fraction < {T.MIN_MOVING_FRACTION} or tracer_density < {T.MIN_TRACER_DENSITY}/m² have nothing to
  track. Every method then returns confident near-zero speeds (-95 to -100 % error). Refuse and advise seeding.
- KLT (feature tracking with filters) is the most accurate method on 9 of 11 rivers. LSPIV covers more points but
  ranges from 17 % to 88 % error. To fill KLT gaps prefer sample_points(fill_with='wide'); filling with LSPIV
  made three rivers worse even after its agreement check. STIV is unreliable except on texture-poor oblique water.
- Coverage of gauge points grows with clip length (Murg 74 % -> 99 % going from 150 to 359 frames). If coverage is
  below 80 % and more frames are available, load more (up to 400).
- camera_motion_px > 1 (measured on the banks): stabilise. stabilise checks itself; if it reports kept=false the
  bank is not rigid, and if the camera still moves more than 3 px, refuse (advise a tripod).
- flow_jitter > {T.FLOW_JITTER_MAX} px/frame means the whole picture jumps between frames (shake), even with no
  bank in view. Use stabilise_from_flow when the bank method is unavailable or failed; if that is not kept, refuse.
  On clean clips flow_jitter is <= 0.84 at all 11 sites; 2 px shake gives >= 2.5.
- load_clip reporting repeated_frame_share > 0.3: call remove_repeated_frames first. A re-encoded file repeats
  frames and every speed would come out 2-3x too high (+100 to +200 %).
- disp_px below {T.TARGET_DISP_PX[0]} px/frame: increase stride so displacement is about 2 px.
- After any repair that reloads frames (load_clip), repeat the repairs; load_clip resets the frames.
- flicker above {T.FLICKER_PREP}: set_preprocess('background').
- Discharge: compute_discharge checks the two-section continuity, coverage and Froude number. If it fails, try one
  remedy that addresses the stated problem (e.g. more frames). If it still fails, escalate. Never publish a failed
  result.
- Points with a high local spread are flagged low-confidence automatically; mention that in your summary.
Be efficient: do not repeat a tool call with identical arguments. End with publish, refuse or escalate, then
give a two-sentence summary for the operator."""


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else round(float(o), 4)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def _site_brief(s: T.Session):
    c = s.case
    b = {"site": c.name, "platform": c.platform, "country": c.country, "fps": c.fps, "gsd_m_per_px": c.gsd,
         "frames_on_disk": c.available(), "water_polygon": bool(c.water),
         "gauge_points": 0 if s.targets is None else len(s.targets),
         "surveyed_sections": list(s.sections) if s.sections else [],
         "flood_threshold_m3s": s.flood_q}
    return json.dumps(b)


def plan(s: T.Session, call, verbose=True, model=MODEL, on_text=None):
    import boto3
    br = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    tools = [{"toolSpec": {"name": t["name"], "description": t["description"],
                           "inputSchema": {"json": t["input_schema"]}}} for t in T.TOOL_SPECS]
    msgs = [{"role": "user", "content": [{"text": "New clip to measure. Site: " + _site_brief(s)}]}]
    s.llm_usage = {"input_tokens": 0, "output_tokens": 0, "turns": 0, "model": model}
    for _ in range(MAX_TURNS):
        r = br.converse(modelId=model, system=[{"text": SYSTEM}], messages=msgs,
                        toolConfig={"tools": tools}, inferenceConfig={"maxTokens": 1500, "temperature": 0})
        u = r.get("usage", {})
        s.llm_usage["input_tokens"] += u.get("inputTokens", 0)
        s.llm_usage["output_tokens"] += u.get("outputTokens", 0)
        s.llm_usage["turns"] += 1
        out = r["output"]["message"]
        msgs.append(out)
        uses = [c["toolUse"] for c in out["content"] if "toolUse" in c]
        for c in out["content"]:
            if "text" in c and verbose:
                print("  claude:", c["text"].strip().replace("\n", " ")[:400])
            if "text" in c and on_text and uses and c["text"].strip():
                on_text(c["text"].strip())
        if not uses:
            s.llm_summary = " ".join(c.get("text", "") for c in out["content"]).strip()
            return
        results = []
        for tu in uses:
            name, args = tu["name"], tu.get("input") or {}
            if name not in T.TOOLS:
                res = {"error": f"unknown tool {name}"}
            else:
                try:
                    res = call(name, **args)
                except TypeError as e:
                    res = {"error": f"bad arguments: {e}"}
            results.append({"toolResult": {"toolUseId": tu["toolUseId"], "content": [{"json": _clean(res)}]}})
        msgs.append({"role": "user", "content": results})
    call("escalate", reason="planner hit the turn limit without a decision")
