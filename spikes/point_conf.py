"""Does local vector agreement predict per-point error? Pools KLT points over all cases."""
import warnings, numpy as np
warnings.filterwarnings("ignore")
from gaugeless.cases import CASES, load_frames
from gaugeless.evaluate import run_method, ALPHA
rows = []
for n in ["noce","bradano","alpine","arrow_uas_seeded","arrow_gopro_seeded","brenta_flir","thalhofen","lavence","murg"]:
    c = CASES[n]; ref = c.reference(); fr = load_frames(c, 400)
    f = run_method("klt", fr, c, ref); r = 0.3/c.gsd; r = np.clip(r, 8, 60)
    for p in ref.itertuples():
        if p.kind == "adcp_top" or p.v <= 0.05: continue
        m = np.hypot(f.pts[:,0]-p.x_px, f.pts[:,1]-p.y_px) <= r
        if m.sum() < 3: continue
        s = f.speed[m]; med = np.median(s)
        truth = p.v / ALPHA if p.kind == "depth_avg" else p.v
        a = np.arctan2(f.vel[m,1], f.vel[m,0]); R = abs(np.mean(np.exp(1j*a)))
        rows.append((n, m.sum(), (np.percentile(s,75)-np.percentile(s,25))/med, R, abs(med/truth-1)))
import pandas as pd
d = pd.DataFrame(rows, columns=["case","n","iqr_rel","dir_R","err"])
d["n_bin"] = pd.qcut(d.n, 4); d["iqr_bin"] = pd.qcut(d.iqr_rel, 4); d["R_bin"] = pd.qcut(d.dir_R, 4, duplicates="drop")
for b in ["n_bin","iqr_bin","R_bin"]:
    print(d.groupby(b, observed=True).err.agg(["count","median","mean"]).round(3))
print(d.groupby("case").err.agg(["count","median","mean"]).round(3))
print("spearman", d[["n","iqr_rel","dir_R","err"]].corr("spearman").err.round(3).to_dict())
d.to_csv("results/point_conf.csv", index=False)
