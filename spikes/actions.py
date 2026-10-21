"""Measure what candidate agent actions actually do to accuracy and coverage (KLT)."""
import sys, warnings, numpy as np
warnings.filterwarnings("ignore")
from gaugeless.cases import CASES, load_frames
from gaugeless.evaluate import run_method, sample, metrics, ALPHA

def score(case, f, radius=0.2, min_n=3):
    ref = case.reference(); est = sample(f, ref, case.gsd, radius_m=radius, min_n=min_n)
    out = {}
    for kind, fac in [("surface",1),("adcp_top",1),("integrated",1),("depth_avg",ALPHA)]:
        m = (ref.kind==kind).to_numpy()
        if m.any() and kind != "adcp_top": out[kind] = metrics(est[m]*fac, ref.v.to_numpy()[m])
    return out

for n in sys.argv[1:]:
    c = CASES[n]; ref = c.reference()
    for nf in [150, 400]:
        fr = load_frames(c, nf)
        f = run_method("klt", fr, c, ref)
        for rad in [0.2, 0.5]:
            s = score(c, f, rad)
            print(f"{n:20s} frames={len(fr):3d} radius={rad} ", {k:(v['n'], round(v['coverage']*100), round(v['mape'] or -1,1)) for k,v in s.items()})
