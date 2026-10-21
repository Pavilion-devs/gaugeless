"""Score velocimetry methods against in-river reference measurements.

    python -m gaugeless.evaluate                     # every case with frames on disk, every method
    python -m gaugeless.evaluate --cases alpine noce --methods klt lspiv

Comparison rules (stated in the report):
  * surface / adcp_top references: compared with the measured surface speed directly.
  * depth_avg references: compared with ALPHA * surface speed (velocity index), reported separately.
  * integrated (Tiber radar box): mean surface speed inside the box.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd

from . import methods as M
from .cases import CASES, load_frames

ALPHA = 0.85


def sample(field: M.VectorField, ref: pd.DataFrame, gsd: float, radius_m=0.2, min_n=3):
    """Per-reference estimate: median speed of vectors within radius (or inside the box)."""
    r_px = float(np.clip(radius_m / gsd, 8, 40))
    out = np.full(len(ref), np.nan)
    if len(field.pts) == 0:
        return out
    s = field.speed
    for k, row in enumerate(ref.itertuples()):
        if row.kind == "integrated":
            m = (np.abs(field.pts[:, 0] - row.x_px) <= row.w / 2) & (np.abs(field.pts[:, 1] - row.y_px) <= row.h / 2)
        else:
            m = np.hypot(field.pts[:, 0] - row.x_px, field.pts[:, 1] - row.y_px) <= r_px
        if m.sum() >= min_n:
            out[k] = np.median(s[m])
    return out


def metrics(est, ref_v):
    valid = ref_v > 0.02                    # points at ~0 m/s (dead water) make relative error meaningless
    est, ref_v = est[valid], ref_v[valid]
    ok = np.isfinite(est)
    n = int(ok.sum())
    if n == 0:
        return {"n": 0, "coverage": 0.0, "mape": None, "bias": None, "rmse": None}
    e, r = est[ok], ref_v[ok]
    rel = (e - r) / r
    return {"n": n, "coverage": n / len(est), "mape": float(np.mean(np.abs(rel)) * 100),
            "bias": float(np.mean(rel) * 100), "rmse": float(np.sqrt(np.mean((e - r) ** 2)))}


def run_method(name, frames, case, ref, params=None):
    p = dict(params or {})
    pts = ref[["x_px", "y_px"]].to_numpy()
    mask = case.water_mask(frames[0].shape)
    if name == "klt":
        f = M.kltiv_filter(M.klt(frames, case.fps, case.gsd, mask=mask, **p))
    elif name == "lspiv":
        f = M.lspiv(frames, case.fps, case.gsd, points=pts, search=p.pop("search", 24), **p)
        f = f.subset(f.q > 0.15)          # drop border-pinned / flat correlation peaks
    elif name == "dis":
        f = M.dis(frames, case.fps, case.gsd, **p)
    elif name == "stiv":
        # direction from a quick KLT pass (the agent supplies this from its own diagnosis)
        d = M.flow_direction(M.kltiv_filter(M.klt(frames[:30], case.fps, case.gsd, mask=mask)))
        f = M.stiv(frames, case.fps, case.gsd, direction=d, points=pts, **p)
    else:
        raise KeyError(name)
    return f


def evaluate(case_name, method, max_frames=150, params=None):
    case = CASES[case_name]
    ref = case.reference()
    frames = load_frames(case, max_frames=max_frames)
    t0 = time.time()
    f = run_method(method, frames, case, ref, params)
    dt = time.time() - t0
    est = sample(f, ref, case.gsd, min_n=1 if method in ("lspiv", "stiv") else 3)
    rows = {}
    for kind, factor in [("surface", 1.0), ("adcp_top", 1.0), ("integrated", 1.0), ("depth_avg", ALPHA)]:
        m = (ref.kind == kind).to_numpy()
        if m.any():
            rows[kind] = metrics(est[m] * factor, ref.v.to_numpy()[m])
    return {"case": case_name, "method": method, "frames": len(frames), "seconds": round(dt, 1),
            "vectors": int(len(f.pts)), **{f"{k}": v for k, v in rows.items()}}


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--methods", nargs="*", default=["klt", "lspiv", "stiv", "dis"])
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--out", default="results/baseline.jsonl")
    a = ap.parse_args()
    names = a.cases or [n for n, c in CASES.items() if c.ref and c.available() >= 20]
    import os
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a") as fh:
        for n in names:
            for m in a.methods:
                try:
                    r = evaluate(n, m, a.max_frames)
                except Exception as e:
                    r = {"case": n, "method": m, "error": f"{type(e).__name__}: {e}"}
                fh.write(json.dumps(r) + "\n"); fh.flush()
                if "error" in r:
                    print(f"{n:22s} {m:6s} ERROR {r['error']}")
                    continue
                for kind in ("surface", "adcp_top", "integrated", "depth_avg"):
                    if kind in r:
                        k = r[kind]
                        mape = "  --  " if k["mape"] is None else f"{k['mape']:5.1f}%"
                        bias = "" if k["bias"] is None else f"bias {k['bias']:+6.1f}%"
                        print(f"{n:22s} {m:6s} {kind:10s} n={k['n']:3d} cov={k['coverage']*100:3.0f}%  "
                              f"MAPE {mape} {bias}  ({r['seconds']}s)")


if __name__ == "__main__":
    main()
