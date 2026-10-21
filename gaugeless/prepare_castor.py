"""Build the Castor case from RAW video: GCP camera solve (with lens distortion) -> water-plane orthorectification
-> our own frames, plus the ADCP reference positions carried onto our grid by registration.

    python -m gaugeless.prepare_castor

Writes data/derived/castor_v1/{Frames/*.png, reference.csv, section.npz, ortho.json}.
Per the dataset readme: the camera's real rate is 20 fps (metadata says 19.7) and the first 50 frames are corrupt.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
import pandas as pd

from . import ortho as O
from .refs import ROOT, load, _sheet, _num

SRC = os.path.join(ROOT, "CastorRiver/20190410/Video1")
OUT = os.path.join(os.path.dirname(ROOT), "derived", "castor_v1")
STAGE = 3.772      # m, readme
FPS = 20.0
RES = 0.02         # m/px of our grid
SKIP = 50


def castor_section(A):
    """ADCP transect of Reference_Castor_002 as a section on our grid: positions, station, depth, depth-avg v."""
    d = _sheet("CastorRiver/20190410/Reference/Reference_Castor_002.xlsx", 0)
    lab = d.iloc[:, :2].astype(str).apply(lambda c: c.str.lower())
    find = lambda pat: next(i for i in range(len(d)) if lab.iloc[i].str.contains(pat).any())
    rx, ry, rdist, rdep = find(r"x \[px\]"), find(r"y \[px\]"), find(r"distance"), find(r"river dep")
    bins = [i for i in range(ry + 2, rdep) if not np.isnan(_num(pd.Series([d.iat[i, 1]]))[0])]
    rows = []
    for j in range(2, d.shape[1]):
        x, y, st, h = _num(pd.Series([d.iat[rx, j], d.iat[ry, j], d.iat[rdist, j], d.iat[rdep, j]]))
        if np.isnan(st) or np.isnan(h):
            continue
        prof = _num(d.iloc[bins, j]).to_numpy()
        rows.append((x, y, st, h, float(np.nanmean(prof)) if np.isfinite(prof).any() else 0.0))
    a = np.array(rows, float)
    a = a[np.argsort(a[:, 2])]
    # bank entries have no pixel position: extrapolate along the transect line from the measured ones
    ok = np.isfinite(a[:, 0])
    for c in (0, 1):
        k = np.polyfit(a[ok, 2], a[ok, c], 1)
        a[~ok, c] = np.polyval(k, a[~ok, 2])
    px = cv2.transform(a[:, :2].astype(np.float32)[None], A)[0]
    q = float(np.trapezoid(a[:, 4] * a[:, 3], a[:, 2]))
    return px, a[:, 2] - a[0, 2], a[:, 3], a[:, 4], q


def main():
    os.makedirs(os.path.join(OUT, "Frames"), exist_ok=True)
    g = pd.read_csv(os.path.join(SRC, "GCPs_use.csv")).to_numpy(float)
    cap = cv2.VideoCapture(os.path.join(SRC, "Video", "Castor_Canada_20190410_112239.mp4"))
    size = (int(cap.get(3)), int(cap.get(4)))
    cam = O.calibrate(g[:, :3], g[:, 3:5], size)
    grid = O.grid_for(g[:, :2], RES, margin=6)
    maps = O.ortho_maps(cam, grid, STAGE)
    n = 0
    i = 0
    first = None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        i += 1
        if i <= SKIP:
            continue
        o = O.remap(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), maps)
        if first is None:
            first = o
        n += 1
        cv2.imwrite(os.path.join(OUT, "Frames", f"{n:05d}.png"), o)
    A, reg = O.register_to(first, cv2.imread(os.path.join(SRC, "Frames", "00051.jpg"), 0), 0.01 / RES)
    ref = load("castor_v1")
    p = cv2.transform(ref[["x_px", "y_px"]].to_numpy(np.float32)[None], A)[0]
    ref = ref.assign(x_px=p[:, 0], y_px=p[:, 1])
    ref.to_csv(os.path.join(OUT, "reference.csv"), index=False)
    px, st, depth, vda, q = castor_section(A)
    np.savez(os.path.join(OUT, "section.npz"), px=px, station=st, depth=depth, v_depth_avg=vda, q_ref=q)
    meta = {"frames": n, "fps": FPS, "gsd": RES, "stage_m": STAGE, "grid": grid.__dict__,
            "camera": {"f_px": float(cam["K"][0, 0]), "k1": float(cam["dist"].ravel()[0]), "rms_px": cam["rms_px"],
                       "loo_median_px": cam["loo_median_px"], "loo_max_px": cam["loo_max_px"]},
            "registration_to_published_grid": reg, "q_adcp": q, "section_width_m": float(st[-1])}
    json.dump(meta, open(os.path.join(OUT, "ortho.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
