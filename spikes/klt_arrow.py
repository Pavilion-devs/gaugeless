"""Day-0 spike: plain KLT (KLT-IV style) on River Arrow UAS footage vs current-meter reference."""
import os, sys, cv2, numpy as np, pandas as pd
root = os.path.expanduser("~/Documents/opencv26/data/perks/RiverArrow")
clip = sys.argv[1] if len(sys.argv) > 1 else "Non-stabilised/UAS_Seeded"
GSD, FPS = 0.0174, 5.0            # m/px, Hz (Readme)
fdir = f"{root}/{clip}/Frames"
frames = [cv2.cvtColor(cv2.imread(f"{fdir}/{f}"), cv2.COLOR_BGR2GRAY) for f in sorted(os.listdir(fdir))]

# reference points (x, y, v) from the spreadsheet
d = pd.read_excel(f"{root}/Reference/Validation_Data_Arrow_UAS_v2.xlsx", header=None)
ref = d[pd.to_numeric(d[0], errors="coerce").notna() & pd.to_numeric(d[2], errors="coerce").notna()]
ref = ref[[0, 1, 2]].astype(float).to_numpy()

lk = dict(winSize=(31, 31), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
pts_all, vel_all = [], []
for i in range(len(frames) - 1):
    a, b = frames[i], frames[i + 1]
    p0 = cv2.goodFeaturesToTrack(a, maxCorners=4000, qualityLevel=0.005, minDistance=4, blockSize=7)
    if p0 is None: continue
    p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None, **lk)
    pb, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, p1, None, **lk)       # forward-backward check
    fb = np.linalg.norm(p0 - pb, axis=2).ravel()
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 0.5)
    disp = (p1 - p0).reshape(-1, 2)[ok]
    v = np.linalg.norm(disp, axis=1) * GSD * FPS
    pts_all.append(p0.reshape(-1, 2)[ok]); vel_all.append(np.c_[v, disp])
P, VV = np.vstack(pts_all), np.vstack(vel_all)
V = VV[:, 0]
MODE = sys.argv[2] if len(sys.argv) > 2 else "naive"
if MODE == "kltiv":   # KLT-IV filters: >=0.1 m/s and within 45 deg of dominant flow direction
    fast = V >= 0.1
    ang = np.arctan2(VV[fast, 2], VV[fast, 1]); flow = np.arctan2(np.median(np.sin(ang)), np.median(np.cos(ang)))
    dang = np.abs(np.angle(np.exp(1j * (np.arctan2(VV[:, 2], VV[:, 1]) - flow))))
    keep = fast & (dang < np.pi / 4); P, V = P[keep], V[keep]
else:
    keep = V > 0.02; P, V = P[keep], V[keep]

R = 12  # px search radius around each reference point
est = []
for x, y, vr in ref:
    m = np.hypot(P[:, 0] - x, P[:, 1] - y) < R
    est.append(np.median(V[m]) if m.sum() >= 5 else np.nan)
est = np.array(est); ok = ~np.isnan(est)
err = (est[ok] - ref[ok, 2]) / ref[ok, 2]
print(f"{clip}: {len(frames)} frames, {len(V)} vectors, {ok.sum()}/{len(ref)} ref points covered")
print(f"MAPE {np.mean(np.abs(err))*100:.1f}%  bias {np.mean(err)*100:+.1f}%  "
      f"RMSE {np.sqrt(np.mean((est[ok]-ref[ok,2])**2)):.3f} m/s  (ref mean {ref[ok,2].mean():.3f} m/s)")
