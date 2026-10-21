"""Reference-velocity loaders for the Perks et al. (2020) image velocimetry dataset.

Every case ships its in-river measurements in a different spreadsheet layout. Each loader
returns one normalised table:

    x_px, y_px  location in the case's reference frame (pixels)
    v           velocity magnitude (m/s)
    kind        what the number physically is:
                  "surface"      measured at / just below the water surface
                  "adcp_top"     shallowest valid ADCP bin in that column (near-surface)
                  "depth_avg"    depth-averaged velocity for the column (ADCP mean, or 0.6-depth point)
                  "integrated"   one value representing a pixel box (x_px, y_px = box centre)
    group       transect / vertical id, for profile plots
    w, h        box size for "integrated" rows, else 0

Surface-velocity methods are scored against "surface" / "adcp_top" directly, and against
"depth_avg" after applying a velocity index alpha (reported separately, never silently mixed).
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

ROOT = os.environ.get("GAUGELESS_DATA") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "perks")
COLS = ["x_px", "y_px", "v", "kind", "group", "w", "h"]


def _sheet(path, sheet=0):
    return pd.read_excel(os.path.join(ROOT, path), sheet_name=sheet, header=None)


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _rows(records):
    df = pd.DataFrame(records, columns=COLS)
    df[["x_px", "y_px", "v", "w", "h"]] = df[["x_px", "y_px", "v", "w", "h"]].astype(float)
    return df.dropna(subset=["x_px", "y_px", "v"]).reset_index(drop=True)


# ---------- layout A: header row "x | y | vel" followed by numeric rows ----------
def point_table(path, kind="surface"):
    """Noce, Bradano, Alpine, Arrow (several XS blocks). Finds every 'x,y,vel' header."""
    d = _sheet(path)
    out, group = [], "pts"
    for i in range(len(d)):
        row = [str(c).strip().lower() for c in d.iloc[i].tolist()]
        first = next((c for c in row if c != "nan"), "")
        if first.startswith("xs") and len(first) <= 4:
            group = first.upper()
        if "x" in row and "y" in row:
            cx, cy = row.index("x"), row.index("y")
            cv = next(j for j, c in enumerate(row) if c.startswith("vel"))
            j = i + 1
            while j < len(d) and not np.isnan(_num(pd.Series([d.iat[j, cx]]))[0]):
                out.append([d.iat[j, cx], d.iat[j, cy], d.iat[j, cv], kind, group, 0, 0])
                j += 1
    return _rows(out)


# ---------- layout A': "X | Y | Vel1..Vel12" replicates (Brenta) ----------
def replicate_table(path, replicate=None):
    """Brenta: 12 replicate current-meter readings per point. replicate=None -> mean."""
    d = _sheet(path)
    hdr = next(i for i in range(len(d)) if "X" in [str(c).strip() for c in d.iloc[i]])
    names = [str(c).strip() for c in d.iloc[hdr]]
    cx, cy = names.index("X"), names.index("Y")
    vcols = [j for j, n in enumerate(names) if n.lower().startswith("vel")]
    out = []
    for i in range(hdr + 1, len(d)):
        x, y = _num(pd.Series([d.iat[i, cx], d.iat[i, cy]]))
        if np.isnan(x):
            continue
        vals = _num(d.iloc[i, vcols]).to_numpy()
        v = np.nanmean(vals) if replicate is None else vals[replicate - 1]
        out.append([x, y, v, "surface", str(d.iat[i, 0]), 0, 0])
    return _rows(out)


# ---------- layout B: ADCP grid (columns = verticals, rows = depth bins) ----------
def adcp_grid(path, sheet=0):
    """Thalhofen, La Vence, Murg, Castor 0410. Row labelled 'x [px]' and 'y [px]', then depth rows."""
    d = _sheet(path, sheet)
    lab = d.iloc[:, :2].astype(str).apply(lambda c: c.str.lower())
    rx = next(i for i in range(len(d)) if lab.iloc[i].str.contains(r"x \[px\]").any())
    ry = next(i for i in range(len(d)) if lab.iloc[i].str.contains(r"y \[px\]").any())
    # depth rows: first column (or second) holds a numeric depth
    depth_col = 0 if _num(d.iloc[ry + 1:, 0]).notna().sum() > _num(d.iloc[ry + 1:, 1]).notna().sum() else 1
    depth_rows = [i for i in range(ry + 1, len(d)) if not np.isnan(_num(pd.Series([d.iat[i, depth_col]]))[0])]
    out = []
    for j in range(2, d.shape[1]):
        x, y = _num(pd.Series([d.iat[rx, j], d.iat[ry, j]]))
        if np.isnan(x) or np.isnan(y):
            continue
        prof = _num(d.iloc[depth_rows, j]).to_numpy()
        if not np.isfinite(prof).any():
            continue
        top = prof[np.isfinite(prof)][0]
        out.append([x, y, top, "adcp_top", f"col{j}", 0, 0])
        out.append([x, y, np.nanmean(prof), "depth_avg", f"col{j}", 0, 0])
    return _rows(out)


# ---------- layout C: FlowTracker ADV sheets (Salmon, Castor 0709) ----------
def flowtracker(path, sheet=0):
    """Rows with station, pixel x/y, %Dep, velocity. %Dep 0 -> surface; 0.6 -> depth-averaged proxy."""
    d = _sheet(path, sheet)
    hdr = [i for i in range(len(d)) if str(d.iat[i, 0]).strip() == "St"]
    out = []
    for h in hdr:
        names = [str(c).strip() for c in d.iloc[h]]
        cdep = names.index("%Dep")
        if "Vel" in names:
            cvel = [names.index("Vel")]
        else:
            cvel = [names.index("VelX"), names.index("VelY")]
        for i in range(h + 2, len(d)):
            st = str(d.iat[i, 0]).strip()
            if st in ("St", "nan") or not st.replace(".", "").isdigit():
                if st in ("St",):
                    break
                continue
            x, y, dep = _num(pd.Series([d.iat[i, 2], d.iat[i, 3], d.iat[i, cdep]]))
            vv = _num(d.iloc[i, cvel]).to_numpy()
            v = float(np.hypot(*vv)) if len(vv) == 2 else float(vv[0])
            if np.isnan(dep) or np.isnan(v) or v == 0:
                continue
            kind = "surface" if dep == 0 else ("depth_avg" if abs(dep - 0.6) < 1e-6 else None)
            if kind:
                out.append([x, y, v, kind, f"st{st}", 0, 0])
    return _rows(out).drop_duplicates(subset=["x_px", "y_px", "kind"])


# ---------- layout D: single integrated value over a box (Tiber surface radar) ----------
def tiber():
    d = _sheet("RiverTiber/Reference/Validation_Data_RiverTiber.xlsx")
    v = next(float(d.iat[i, 1]) for i in range(len(d)) if str(d.iat[i, 0]).strip() == "Vel")
    xs, ys = [], []
    for i in range(len(d)):
        if str(d.iat[i, 8]).strip() in "ABCD" and str(d.iat[i, 8]).strip():
            x, y = _num(pd.Series([d.iat[i, 9], d.iat[i, 10]]))
            if not np.isnan(x):
                xs.append(x); ys.append(y)
    return _rows([[np.mean(xs), np.mean(ys), v, "integrated", "SVR", max(xs) - min(xs), max(ys) - min(ys)]])


LOADERS = {
    "noce": lambda: point_table("RiverNoce/Reference/Validation_Data_Noce.xlsx"),
    "bradano": lambda: point_table("RiverBradano/Reference/Validation_Data_RiverBradano.xlsx"),
    "alpine": lambda: point_table("AlpineRiver/Reference/Validation_Data_RiverAlpine.xlsx"),
    "arrow_uas": lambda: point_table("RiverArrow/Reference/Validation_Data_Arrow_UAS_v2.xlsx"),
    "arrow_gopro": lambda: point_table("RiverArrow/Reference/Validation_Data_Arrow_GoPro.xlsx"),
    "brenta_flir": lambda: replicate_table("RiverBrenta/FLIR/Reference/Validation_Data_RiverBrenta_FLIR.xlsx"),
    "brenta_gopro": lambda: replicate_table("RiverBrenta/GoPro/Reference/Validation_Data_RiverBrenta_GoPro.xlsx"),
    "thalhofen": lambda: adcp_grid("RiverThalhofen/Reference/Thalhofen_Transect_0.xlsx"),
    "lavence": lambda: adcp_grid("RiverLaVence/Reference/Vence_Transect_0.xlsx"),
    "murg": lambda: adcp_grid("MurgRiver/Reference/Reference_Data_Murg.xlsx"),
    "castor_v1": lambda: adcp_grid("CastorRiver/20190410/Reference/Reference_Castor_002.xlsx"),
    "castor_v4": lambda: flowtracker("CastorRiver/20190709/Reference/02LB006_20190709_velocity validation.xlsx"),
    "salmon": lambda: flowtracker("SalmonRiver/Reference/08HD006_20190604_velocity validation.xlsx",
                                  "velocity validation"),
    "castor": lambda: pd.read_csv(os.path.join(ROOT, "..", "derived", "castor_v1", "reference.csv")),
    "tiber": tiber,
}


def load(case: str) -> pd.DataFrame:
    return LOADERS[case]()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    for name in LOADERS:
        try:
            r = load(name)
            k = r.groupby("kind")["v"].agg(["count", "mean"]).round(3)
            print(f"{name:13s} " + "  ".join(f"{i}: n={row['count']:.0f} mean={row['mean']:.3f}"
                                             for i, row in k.iterrows()))
        except Exception as e:
            print(f"{name:13s} ERROR {type(e).__name__}: {e}")


def lamorge_sections():
    """Two surveyed cross-sections in the La Morge view (pixel positions + depth) and the gauged discharge.

    Discharge: current-meter mid-section gauging 2018-10-11, stage 1.10 m, Q = 1.321 m³/s, A = 3.41 m²
    (sheet 'ENG_example' of streamgauging_sheet_midsection.xlsx, made ~5 m downstream of the view).
    """
    import pandas as pd
    base = os.path.join(ROOT, "LaMorgeRiver/Reference")
    x = pd.read_excel(os.path.join(base, "Cross-Section.xlsx"), header=None)
    secs = {}
    for name, (cd, cx, cy) in {"right": (4, 5, 6), "left": (14, 15, 16)}.items():
        rows = x.iloc[2:32, [cd, cx, cy]].apply(pd.to_numeric, errors="coerce").dropna()
        secs[name] = (rows.iloc[:, 1:].to_numpy(float), rows.iloc[:, 0].to_numpy(float))
    g = pd.read_excel(os.path.join(base, "streamgauging_sheet_midsection.xlsx"), sheet_name="ENG_example", header=None)
    q = float(g.iloc[8, 11]); area = float(g.iloc[8, 12])
    return secs, {"q": q, "area": area, "stage": 1.1}


ADCP_SECTIONS = {
    "thalhofen": ("RiverThalhofen/Reference/Thalhofen_Transect_0.xlsx", None),
    "lavence": ("RiverLaVence/Reference/Vence_Transect_0.xlsx", None),
    "murg": ("MurgRiver/Reference/Reference_Data_Murg.xlsx", 2.76),     # Q stated in the sheet
}


def adcp_section(key):
    """An ADCP transect as a cross-section: pixel position, station (m), river depth (m) and depth-averaged
    velocity per vertical, plus the reference discharge (stated in the sheet, or integrated from the ADCP)."""
    path, q_stated = ADCP_SECTIONS[key]
    d = _sheet(path, 0)
    lab = d.iloc[:, :2].astype(str).apply(lambda c: c.str.lower())
    find = lambda pat: next(i for i in range(len(d)) if lab.iloc[i].str.contains(pat).any())
    rx, ry, rdepth = find(r"x \[px\]"), find(r"y \[px\]"), find(r"river dep")
    rdist = next(i for i in range(len(d)) if lab.iloc[i].str.contains(r"dist").any())
    depth_col = 0 if _num(d.iloc[ry + 1:rdepth, 0]).notna().sum() > _num(d.iloc[ry + 1:rdepth, 1]).notna().sum() else 1
    bins = [i for i in range(ry + 1, rdepth) if not np.isnan(_num(pd.Series([d.iat[i, depth_col]]))[0])]
    rows = []
    for j in range(2, d.shape[1]):
        x, y, st, h = _num(pd.Series([d.iat[rx, j], d.iat[ry, j], d.iat[rdist, j], d.iat[rdepth, j]]))
        if np.isnan(x) or np.isnan(y) or np.isnan(st) or np.isnan(h):
            continue
        prof = _num(d.iloc[bins, j]).to_numpy()
        v = float(np.nanmean(prof)) if np.isfinite(prof).any() else 0.0
        rows.append((x, y, st, h, v))
    a = np.array(rows, float)
    a = a[np.argsort(a[:, 2])]
    q_adcp = float(np.trapezoid(a[:, 4] * a[:, 3], a[:, 2]))
    return {"px": a[:, :2], "station": a[:, 2] - a[0, 2], "depth": a[:, 3], "v_depth_avg": a[:, 4],
            "q_ref": q_stated if q_stated else q_adcp, "q_adcp": q_adcp,
            "q_source": "stated in reference sheet" if q_stated else "integrated from ADCP verticals"}
