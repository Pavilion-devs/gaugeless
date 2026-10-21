"""Run a planner on a case and score what it published.

    python -m gaugeless.agent.run lamorge noce --planner rules
    python -m gaugeless.agent.run --all --out results/agent_rules.jsonl

The planner gets the gauge-point *positions* (where the operator wants velocities) and, where a site has them,
surveyed cross-sections. Reference velocities are only used afterwards, for scoring.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np

import os

from ..cases import CASES
from ..refs import ROOT
from ..evaluate import metrics, ALPHA
from . import tools as T


def make_session(name: str, flood_q=None) -> T.Session:
    c = CASES[name]
    s = T.Session(case=c, flood_q=flood_q)
    ref = c.reference()
    if ref is not None:
        s.targets = ref[["x_px", "y_px"]].to_numpy(float)
    if name == "lamorge":
        from ..refs import lamorge_sections
        from ..discharge import Section
        secs, _ = lamorge_sections()
        s.sections = {k: Section(k, p, d, c.gsd) for k, (p, d) in secs.items()}
    elif name == "castor":
        from ..discharge import Section
        z = np.load(os.path.join(ROOT, "..", "derived", "castor_v1", "section.npz"))
        s.sections = {"adcp": Section("adcp", z["px"], z["depth"], c.gsd, z["station"])}
    elif name in ADCP_SITES:
        from ..refs import adcp_section
        from ..discharge import Section
        a = adcp_section(name)
        s.sections = {"adcp": Section("adcp", a["px"], a["depth"], c.gsd, a["station"])}
    return s


ADCP_SITES = ("thalhofen", "lavence", "murg")


def q_reference(name):
    if name == "lamorge":
        from ..refs import lamorge_sections
        return lamorge_sections()[1]["q"]
    if name in ADCP_SITES:
        from ..refs import adcp_section
        return adcp_section(name)["q_ref"]
    if name == "castor":
        return float(np.load(os.path.join(ROOT, "..", "derived", "castor_v1", "section.npz"))["q_ref"])
    return None


def run(name: str, planner: str = "rules", flood_q=None, verbose=True):
    s = make_session(name, flood_q)

    def call(tool, **args):
        r = T.TOOLS[tool](s, **args)
        if verbose:
            short = {k: v for k, v in r.items() if k not in ("sections",)} if isinstance(r, dict) else r
            print(f"  [{len(s.trace):2d}] {tool}({', '.join(f'{k}={v}' for k, v in args.items())}) -> {short}")
        return r

    t0 = time.time()
    if planner == "rules":
        from . import rules
        rules.plan(s, call)
    elif planner == "claude":
        from . import llm
        llm.plan(s, call, verbose=verbose)
    elif planner == "local":
        from . import local
        local.plan(s, call, verbose=verbose)
    else:
        raise KeyError(planner)
    return s, time.time() - t0


def score(name: str, s: T.Session):
    out = {"case": name, "outcome": "published" if s.published else ("refused" if s.refused else
                                                                     ("escalated" if s.escalated else "none")),
           "steps": len(s.trace)}
    c = CASES[name]
    ref = c.reference()
    if ref is not None and s.point_results is not None:
        v = s.point_results["v"]
        for kind, fac in [("surface", 1.0), ("integrated", 1.0), ("depth_avg", ALPHA)]:
            m = (ref.kind == kind).to_numpy()
            if m.any():
                out[kind] = metrics(v[m] * fac, ref.v.to_numpy()[m])
                hi = m & ~s.point_results["low_conf"]
                out[kind + "_highconf"] = metrics(v[hi] * fac, ref.v.to_numpy()[hi])
    if s.discharge is not None:
        out["q"] = s.discharge["q"]
        out["q_passed"] = s.discharge["passed"]
        qref = q_reference(name)
        if qref:
            out["q_ref"] = round(qref, 4)
            out["q_err"] = round(s.discharge["q"] / qref - 1, 4)
    return out


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--planner", default="rules")
    ap.add_argument("--out")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    names = a.cases or [n for n, c in CASES.items() if (c.ref or n == "lamorge") and n != "alpine_full"]
    for n in names:
        print(f"== {n} ({a.planner})")
        s, dt = run(n, a.planner, verbose=not a.quiet)
        r = score(n, s)
        r["planner"], r["seconds"] = a.planner, round(dt, 1)
        r["llm_usage"] = getattr(s, "llm_usage", None)
        r["trace"] = [{k: v for k, v in st.items() if k != "result"} | {"result_keys": list(st["result"])}
                      for st in s.trace]
        brief = {k: (round(v["mape"], 1) if isinstance(v, dict) and v.get("mape") is not None else v)
                 for k, v in r.items() if k not in ("trace",)}
        print("  =>", brief)
        if a.out:
            with open(a.out, "a") as fh:
                fh.write(json.dumps(r, default=float) + "\n")


if __name__ == "__main__":
    main()
