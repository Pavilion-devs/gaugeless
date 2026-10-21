"""Per-stage benchmark: stock OpenCV 5 vs COOL on the same machine, same clips. Results carry provenance.

    python -m bench.bench --out results/bench_$(hostname).json [--qwen models/qwen2.5-0.5b]

Stages: KLT (Shi-Tomasi + PyrLK), LSPIV (matchTemplate ensemble), STIV (remap + FFT), DIS dense flow,
stabilise (PyrLK + RANSAC + warpAffine), full agent run. Optional: Qwen2.5-0.5B prefill tokens/s in cv2.dnn.
Each stage is run `--repeat` times; the median is reported. The $/river-hour line uses the instance's
on-demand price if given (--usd-per-hour).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
import warnings

import cv2
import numpy as np


def provenance():
    info = cv2.getBuildInformation()
    p = {"opencv": cv2.__version__, "opencv_file": cv2.__file__, "build_info_sha1": hashlib.sha1(info.encode()).hexdigest()[:12],
         "machine": platform.machine(), "processor": platform.processor(), "python": platform.python_version(),
         "threads": cv2.getNumThreads(), "kleidicv": "KleidiCV" in info or "kleidicv" in info.lower(),
         "cool": "/opt/cool" in cv2.__file__, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:   # EC2 instance identity (IMDSv2), when on AWS
        import urllib.request
        tok = urllib.request.urlopen(urllib.request.Request("http://169.254.169.254/latest/api/token", method="PUT",
                                     headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"}), timeout=1).read().decode()
        get = lambda k: urllib.request.urlopen(urllib.request.Request(f"http://169.254.169.254/latest/meta-data/{k}",
                                               headers={"X-aws-ec2-metadata-token": tok}), timeout=1).read().decode()
        p |= {"instance_type": get("instance-type"), "instance_id": get("instance-id"), "ami_id": get("ami-id")}
    except Exception:
        pass
    return p


def timed(fn, repeat):
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=["arrow_uas_seeded", "lamorge", "bradano"])
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--qwen")
    ap.add_argument("--usd-per-hour", type=float)
    ap.add_argument("--out", default="results/bench.json")
    a = ap.parse_args()

    from gaugeless import methods as M
    from gaugeless.cases import CASES, load_frames
    from gaugeless.agent import tools as T
    from gaugeless.agent.run import make_session
    from gaugeless.agent import rules

    res = {"provenance": provenance(), "stages": []}
    for name in a.cases:
        c = CASES[name]
        fr = load_frames(c, a.frames)
        h, w = fr[0].shape
        mask = c.water_mask(fr[0].shape)
        pts = np.array([[w * x, h * y] for x in (0.3, 0.5, 0.7) for y in (0.3, 0.5, 0.7)], float)
        n = len(fr)
        stages = {
            "klt": lambda: M.klt(fr, c.fps, c.gsd, mask=mask),
            "lspiv": lambda: M.lspiv(fr, c.fps, c.gsd, points=pts, search=24),
            "stiv": lambda: M.stiv(fr, c.fps, c.gsd, direction=0.0, points=pts),
            "dis": lambda: M.dis(fr, c.fps, c.gsd),
        }
        s = make_session(name); T.load_clip(s, a.frames)
        if s.mask is not None:
            stages["stabilise"] = lambda: T.stabilise(s)
        for st, fn in stages.items():
            sec = timed(fn, a.repeat)
            row = {"case": name, "stage": st, "frames": n, "size": f"{w}x{h}", "seconds": round(sec, 4),
                   "ms_per_frame": round(1000 * sec / n, 3), "x_realtime": round((n / c.fps) / sec, 2)}
            res["stages"].append(row)
            print(json.dumps(row))

        def agent():
            s2 = make_session(name)
            rules.plan(s2, lambda t, **k: T.TOOLS[t](s2, **k))
        sec = timed(agent, 1)
        nf = c.available()
        row = {"case": name, "stage": "agent_end_to_end", "seconds": round(sec, 3),
               "video_seconds": round(min(nf, 400) / c.fps, 1), "x_realtime": round(min(nf, 400) / c.fps / sec, 2)}
        if a.usd_per_hour:
            row["usd_per_river_hour"] = round(a.usd_per_hour * sec / (min(nf, 400) / c.fps), 4)
        res["stages"].append(row)
        print(json.dumps(row))

    if a.qwen:
        from gaugeless.agent.local import Qwen
        q = Qwen(a.qwen)
        ctx = {"moving_fraction": 0.45, "tracer_density": 6.8, "disp_px": 9.9, "flicker": 0.002, "glare_fraction": 0.0,
               "camera_motion_px": 0.1, "repeated_frame_share": 0.0}
        opts = ["stabilise the camera", "remove repeated frames", "mask sun glare", "normalise flicker", "measure now", "refuse"]
        q.choose(ctx, opts)                       # warm-up
        q.calls = q.tokens = 0; q.seconds = 0.0
        for _ in range(a.repeat):
            q.choose(ctx, opts)
        row = {"stage": "qwen_prefill", "load_seconds": round(q.load_s, 2), "prompt_tokens": q.tokens // q.calls,
               "ms_per_decision": round(1000 * q.seconds / q.calls, 1), "tokens_per_s": round(q.tokens / q.seconds, 1)}
        res["stages"].append(row)
        print(json.dumps(row))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
