"""Summarise a degradation run: agent vs fixed on the same points, outcomes, silent failures.

    python -m gaugeless.agent.summarise results/degrade_v2.jsonl
"""
import json
import sys

import numpy as np

from ..cases import CASES
from ..evaluate import ALPHA

SILENT = 40.0


def same_point_err(case, fixed_v, agent_v):
    ref = CASES[case].reference()
    fac = np.where(ref.kind == "depth_avg", ALPHA, 1.0)
    keep = ((ref.kind != "adcp_top") & (ref.v > 0.02)).to_numpy()
    f = np.array(fixed_v, float) * fac; a = np.array(agent_v, float) * fac; r = ref.v.to_numpy()
    both = keep & np.isfinite(f) & np.isfinite(a)
    if not both.any():
        return None, None
    return (float(np.mean(np.abs(f[both] / r[both] - 1)) * 100), float(np.mean(np.abs(a[both] / r[both] - 1)) * 100))


def main(*paths):
    rows = {}
    for path in paths:                       # later files replace earlier rows for the same (case, damage, level)
        for l in open(path):
            r = json.loads(l); rows[(r["case"], r["degradation"], r["level"])] = r
    rows = list(rows.values())
    clean = {r["case"]: r for r in rows if r["degradation"] == "clean"}
    print(f"{'case':18s} {'damage':9s} {'lvl':>5s}  {'fixed':>7s} {'agent':>7s}  same-pts f/a   outcome  repairs")
    # damage-induced silent failure: published, error above 40 % AND 15 points worse than the clean clip
    agg = {"runs": 0, "fixed_silent": 0, "agent_silent": 0, "agent_refused": 0, "agent_escalated": 0,
           "agent_better": 0, "agent_worse": 0, "agent_same": 0}
    def silent(r, side):
        e = r[side]["err"]; c = clean[r["case"]][side]["err"]
        return (r[side]["outcome"] == "published" and e is not None and c is not None and e > 40 and e > c + 15)
    for r in rows:
        fx, ag = r["fixed"], r["agent"]
        sp = same_point_err(r["case"], fx["v"], ag["v"]) if "v" in fx and "v" in ag else (None, None)
        agg["runs"] += 1
        agg["fixed_silent"] += silent(r, "fixed")
        agg["agent_silent"] += silent(r, "agent")
        agg["agent_refused"] += ag["outcome"] == "refused"
        agg["agent_escalated"] += ag["outcome"] == "escalated"
        fe = sp[0] if sp[0] is not None else fx["err"]; ae = sp[1] if sp[1] is not None else ag["err"]
        if fe is not None and ae is not None:
            agg["agent_better"] += ae < fe - 2
            agg["agent_worse"] += ae > fe + 2
            agg["agent_same"] += abs(ae - fe) <= 2
        rep = ",".join(t for t in ag.get("tools", []) if t in ("stabilise", "stabilise_from_flow", "remove_repeated_frames", "mask_glare", "set_preprocess", "refuse", "escalate"))
        fmt = lambda x: "   --  " if x is None else f"{x:6.1f}%"
        sps = "" if sp[0] is None else f"{sp[0]:5.1f}/{sp[1]:5.1f}"
        print(f"{r['case']:18s} {r['degradation']:9s} {str(r['level']):>5s}  {fmt(fx['err'])} {fmt(ag['err'])}  {sps:13s} {ag['outcome'][:8]:8s} {rep}")
    print()
    print(json.dumps(agg))


if __name__ == "__main__":
    main(*sys.argv[1:])
