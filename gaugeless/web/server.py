"""Web endpoint: pick a site (optionally degrade the clip), watch the agent decide step by step, see the result.

    python -m gaugeless.web.server --port 8080

Stdlib only (http.server + Server-Sent Events). Human approval is a separate operator action (/api/approve),
never something the agent can call.
"""
from __future__ import annotations

import argparse
import base64
import functools
import json
import os
import queue
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from .. import degrade as G
from ..agent import tools as T
from ..agent.llm import _clean
from ..agent.run import make_session, score
from ..cases import CASES, load_frames

HERE = os.path.dirname(__file__)
RUNS: dict[str, dict] = {}


def _speed_colour(v, vmax):
    x = np.uint8(np.clip(v / max(vmax, 1e-6), 0, 1) * 255)
    return tuple(int(c) for c in cv2.applyColorMap(np.array([[x]], np.uint8), cv2.COLORMAP_TURBO)[0, 0])


def overlay(s: T.Session, width=900):
    """First frame + velocity vectors + gauge points (white ring = high confidence) + sections."""
    im = load_frames(s.case, 1, color=True)[0] if s.case.available() else None
    if im is None:
        return None
    if s.degrade is not None and s.frames:
        im = cv2.cvtColor(s.frames[0], cv2.COLOR_GRAY2BGR)
    sc = width / im.shape[1]
    im = cv2.resize(im, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    im = (im * 0.6).astype(np.uint8)
    if s.case.water:
        cv2.polylines(im, [(np.array(s.case.water) * sc).astype(np.int32)], True, (255, 200, 80), 1, cv2.LINE_AA)
    f = s.fields.get("klt")
    vmax = 1.0
    if f is not None and len(f.pts):
        vmax = float(np.percentile(f.speed, 95))
        idx = np.random.default_rng(0).choice(len(f.pts), min(1500, len(f.pts)), replace=False)
        fps = T._fps(s)
        for i in idx:
            p = f.pts[i] * sc
            d = f.vel[i] / s.case.gsd / fps * sc * 4          # arrows exaggerated x4
            cv2.arrowedLine(im, tuple(p.astype(int)), tuple((p + d).astype(int)), _speed_colour(f.speed[i], vmax), 1,
                            cv2.LINE_AA, tipLength=0.3)
    if s.sections:
        for k, sec in s.sections.items():
            cv2.polylines(im, [(sec.px * sc).astype(np.int32)], False, (0, 255, 255), 2, cv2.LINE_AA)
            if s.discharge:
                q = s.discharge["sections"][k]["q"]
                p = (sec.px[0] * sc).astype(int)
                cv2.putText(im, f"{k}: {q:.2f} m3/s", (p[0] + 6, max(14, p[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 1, cv2.LINE_AA)
    if s.point_results is not None and s.targets is not None:
        pr = s.point_results
        for p, v, lo in zip(s.targets * sc, pr["v"], pr["low_conf"]):
            c = (90, 90, 90) if not np.isfinite(v) else _speed_colour(v, vmax)
            cv2.circle(im, tuple(p.astype(int)), 5, c, -1, cv2.LINE_AA)
            cv2.circle(im, tuple(p.astype(int)), 6, (255, 255, 255) if not lo else (60, 60, 200), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def start_run(site, planner="rules", degradation=None, level=None, flood_q=None):
    rid = uuid.uuid4().hex[:10]
    q: queue.Queue = queue.Queue()
    run = {"id": rid, "site": site, "planner": planner, "queue": q, "session": None, "done": False}
    RUNS[rid] = run

    def work():
        s = make_session(site, flood_q)
        run["session"] = s
        if degradation and degradation != "none":
            s.degrade = functools.partial(G.apply, name=degradation, level=float(level))
            if degradation == "duplicate":
                s.fps_scale = float(level)

        def call(tool, **args):
            q.put({"type": "call", "tool": tool, "args": args})
            r = T.TOOLS[tool](s, **args)
            step = s.trace[-1]
            q.put({"type": "step", "step": step["step"], "tool": tool, "args": args, "ms": step["ms"],
                   "result": {k: v for k, v in r.items() if k not in ("sections",)} if isinstance(r, dict) else r,
                   "sections": r.get("sections") if isinstance(r, dict) else None})
            if tool in ("measure_velocity", "sample_points", "compute_discharge", "stabilise"):
                q.put({"type": "overlay", "image": overlay(s)})
            return r
        try:
            if planner == "claude":
                from ..agent import llm
                llm.plan(s, call, verbose=False)
                if getattr(s, "llm_summary", None):
                    q.put({"type": "summary", "text": s.llm_summary})
            else:
                from ..agent import rules
                rules.plan(s, call)
            sc = score(site, s)
            q.put({"type": "done", "score": sc, "overlay": overlay(s),
                   "outcome": sc["outcome"], "escalated": s.escalated, "refused": s.refused})
        except Exception as e:
            q.put({"type": "error", "error": f"{type(e).__name__}: {e}"})
        run["done"] = True
        q.put(None)

    threading.Thread(target=work, daemon=True).start()
    return rid


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(_clean(obj), default=str).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        qs = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path in ("/", "/index.html"):
            b = open(os.path.join(HERE, "index.html"), "rb").read()
            self.send_response(200)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif u.path == "/api/sites":
            out = []
            for n, c in CASES.items():
                if n == "alpine_full" or c.available() == 0:
                    continue
                out.append({"site": n, "platform": c.platform, "country": c.country, "fps": c.fps, "gsd": c.gsd,
                            "frames": c.available(), "notes": c.notes, "discharge": n == "lamorge",
                            "reference": bool(c.ref)})
            self._json({"sites": out, "degradations": {k: v[2] for k, v in G.DEGRADATIONS.items()}})
        elif u.path == "/api/run":
            rid = start_run(qs.get("site", "lamorge"), qs.get("planner", "rules"), qs.get("degradation"),
                            qs.get("level"), float(qs["flood_q"]) if qs.get("flood_q") else None)
            self._json({"run_id": rid})
        elif u.path == "/api/stream":
            run = RUNS.get(qs.get("run_id", ""))
            if not run:
                return self._json({"error": "unknown run"}, 404)
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("cache-control", "no-cache")
            self.end_headers()
            while True:
                ev = run["queue"].get()
                if ev is None:
                    break
                self.wfile.write(f"data: {json.dumps(_clean(ev), default=str)}\n\n".encode())
                self.wfile.flush()
        elif u.path == "/api/approve":
            run = RUNS.get(qs.get("run_id", ""))
            if not run or not run["session"]:
                return self._json({"error": "unknown run"}, 404)
            s = run["session"]
            s.approved = True
            s.trace.append({"step": len(s.trace) + 1, "tool": "human_approval", "args": {"by": qs.get("by", "operator")},
                            "result": {"approved": True}, "ms": 0})
            s.refused = None if s.refused and s.refused.startswith("flood") else s.refused
            res = T.publish(s)
            self._json({"publish": res})
        else:
            self._json({"error": "not found"}, 404)


def main():
    import warnings
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    print(f"Gaugeless on http://{a.host}:{a.port}")
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
