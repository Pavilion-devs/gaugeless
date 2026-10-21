"""Agent vs fixed pipeline on controlled degradations of real clips.

    python -m gaugeless.agent.degrade_eval --cases arrow_uas_seeded lavence --out results/degrade.jsonl

fixed:  150 frames, KLT + KLT-IV filters inside the water polygon, always publishes (what a script does).
agent:  the rules planner with all tools; it may repair, extend, refuse or escalate.

Scoring per run: error vs the in-river reference (point MAPE, or |Q error| for La Morge), and outcome.
A *silent failure* is a published result with error > 40 %.
"""
from __future__ import annotations

import argparse
import functools
import json
import time
import warnings

import numpy as np

from .. import degrade as G
from ..cases import CASES, load_frames
from ..evaluate import ALPHA, metrics
from . import tools as T
from .run import make_session, score

SILENT = 40.0


def _point_err(name, v):
    ref = CASES[name].reference()
    fac = np.where(ref.kind == "depth_avg", ALPHA, 1.0)
    keep = (ref.kind != "adcp_top").to_numpy()
    m = metrics((v * fac)[keep], ref.v.to_numpy()[keep])
    return m["mape"], m["coverage"]


def _session(name, deg, d, lv):
    s = make_session(name)
    s.degrade = deg
    if d == "duplicate":
        s.fps_scale = lv          # the re-encoded file claims k x the capture rate
    return s


def fixed(name, deg, d=None, lv=None):
    s = _session(name, deg, d, lv)
    T.load_clip(s, max_frames=150)
    s.diagnosis = None                      # a fixed script does not diagnose
    T.measure_velocity(s, "klt")
    q_err = None
    if s.sections:
        from .run import q_reference
        q = T.compute_discharge(s, "klt")
        q_err = abs(q["q"] / q_reference(name) - 1) * 100
    if s.targets is None:                     # La Morge: discharge is the only reference
        return {"err": q_err, "coverage": None, "outcome": "published", "q_err": q_err}
    T.sample_points(s, "klt")
    e, cov = _point_err(name, s.point_results["v"])
    return {"err": e, "coverage": cov, "outcome": "published", "v": s.point_results["v"].tolist(), "q_err": q_err}


def agent(name, deg, d=None, lv=None):
    from . import rules
    s = _session(name, deg, d, lv)
    rules.plan(s, lambda tool, **a: T.TOOLS[tool](s, **a))
    r = score(name, s)
    if name == "lamorge":
        err = abs(r["q_err"]) * 100 if "q_err" in r else None
        cov = None
    elif s.point_results is not None:
        err, cov = _point_err(name, s.point_results["v"])
    else:
        err, cov = None, None
    out = {"err": err, "coverage": cov, "outcome": r["outcome"], "tools": [t["tool"] for t in s.trace],
           "q_err": abs(r["q_err"]) * 100 if "q_err" in r else None}
    if s.point_results is not None and name != "lamorge":
        out["v"] = s.point_results["v"].tolist()
        out["low_conf"] = s.point_results["low_conf"].tolist()
    return out


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=["arrow_uas_seeded", "lavence", "thalhofen", "noce", "lamorge"])
    ap.add_argument("--degradations", nargs="*", default=list(G.DEGRADATIONS))
    ap.add_argument("--out", default="results/degrade.jsonl")
    a = ap.parse_args()
    for name in a.cases:
        runs = [("clean", 0)] + [(d, lv) for d in a.degradations for lv in G.DEGRADATIONS[d][2]]
        for d, lv in runs:
            deg = None if d == "clean" else functools.partial(G.apply, name=d, level=lv)
            t0 = time.time()
            fx, ag = fixed(name, deg, d, lv), agent(name, deg, d, lv)
            row = {"case": name, "degradation": d, "level": lv, "fixed": fx, "agent": ag, "seconds": round(time.time() - t0, 1)}
            with open(a.out, "a") as fh:
                fh.write(json.dumps(row, default=float) + "\n")
            f_ = lambda r: ("  --  " if r["err"] is None else f"{r['err']:5.1f}%") + f" {r['outcome'][:4]}"
            print(f"{name:18s} {d:9s} {lv:<5} fixed {f_(fx)} | agent {f_(ag)}  "
                  f"{','.join(t for t in ag['tools'] if t not in ('load_clip', 'diagnose_clip', 'measure_velocity', 'sample_points'))}",
                  flush=True)


if __name__ == "__main__":
    main()
