"""Surface velocities -> discharge (m³/s), with uncertainty and a continuity check.

Velocity-area (mid-section) method, as in ISO 748 and the image-velocimetry literature:

    Q = Σ_i  α · u_n,i · h_i · Δw_i

    u_n   surface velocity component normal to the section, sampled from the vector field
    h     flow depth from the bed survey
    α     velocity index (depth-averaged / surface), default 0.85 ± 0.07 (Perks et al. 2025: 0.87 ± 0.07)

Verticals with no trustworthy surface velocity are filled with u ∝ h^(2/3) (Manning scaling) fitted to the
verticals that were measured; the share of discharge that came from filled verticals is reported, because
that is what a hydrologist checks first.

Continuity: two sections in the same reach with no tributary between them must carry the same Q.
|Q1 - Q2| / mean(Q) is a check that needs no reference data at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import methods as M

ALPHA, ALPHA_SD = 0.85, 0.07
trapezoid = getattr(np, "trapezoid", None) or np.trapz   # numpy < 2 (COOL's Python env ships numpy 1.26)


@dataclass
class Section:
    name: str
    px: np.ndarray        # (N, 2) vertical positions in frame pixels, bank to bank
    depth: np.ndarray     # (N,) flow depth, m
    gsd: float
    stations: np.ndarray | None = None   # surveyed distance along the section, m (else from pixels x gsd)

    @property
    def station(self):     # distance along the section, m
        if self.stations is not None:
            return np.asarray(self.stations, float)
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(self.px, axis=0), axis=1))]
        return d * self.gsd

    @property
    def normal(self):      # unit normal (image axes) of the straight line bank to bank
        t = self.px[-1] - self.px[0]
        t = t / np.linalg.norm(t)
        return np.array([-t[1], t[0]])

    @property
    def area(self):
        return float(trapezoid(self.depth, self.station))


@dataclass
class DischargeResult:
    section: str
    q: float               # m³/s
    q_sd: float            # one standard deviation, m³/s
    area: float
    mean_velocity: float
    measured_share: float  # share of Q from verticals with a measured surface velocity
    alpha: float
    verticals: list = field(default_factory=list)

    def as_dict(self):
        d = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items() if k != "verticals"}
        return d


def sample_normal(fld: M.VectorField, sec: Section, radius_m=0.25, min_n=3):
    """Median normal velocity at each vertical, NaN where too few vectors."""
    n = sec.normal
    r = radius_m / sec.gsd
    out = np.full(len(sec.px), np.nan)
    if len(fld.pts) == 0:
        return out
    un = fld.vel @ n
    for i, p in enumerate(sec.px):
        m = np.hypot(*(fld.pts - p).T) <= r
        if m.sum() >= min_n:
            out[i] = np.median(un[m])
    return out


def discharge(fld: M.VectorField, sec: Section, alpha=ALPHA, alpha_sd=ALPHA_SD, radius_m=0.25, min_n=3):
    u = sample_normal(fld, sec, radius_m, min_n)
    if np.nanmedian(u) < 0:                     # section drawn the other way round
        u = -u
    h = np.clip(sec.depth, 0, None)
    measured = np.isfinite(u) & (h > 0)
    w23 = h ** (2 / 3)
    if measured.sum() >= 2:
        k = np.median(u[measured] / np.maximum(w23[measured], 1e-6))   # Manning scaling coefficient
    elif measured.sum() == 1:
        k = float(u[measured][0] / max(w23[measured][0], 1e-6))
    else:
        k = np.nan
    u_full = np.where(measured, u, k * w23)
    x = sec.station
    q_unit = alpha * u_full * h                  # m²/s per vertical
    q = float(trapezoid(q_unit, x))
    q_meas = float(trapezoid(np.where(measured, q_unit, 0), x))
    share = q_meas / q if q > 0 else 0.0
    # uncertainty: α dominates; filled verticals add ±30 % of their contribution; sampling spread of measured ones
    filled = q - q_meas
    var = (q * alpha_sd / alpha) ** 2 + (0.3 * filled) ** 2
    q_sd = float(np.sqrt(var))
    area = sec.area
    verts = [dict(station=round(float(s), 3), depth=round(float(d), 3),
                  u_surface=(round(float(v), 3) if np.isfinite(v) else None), measured=bool(m))
             for s, d, v, m in zip(x, h, u, measured)]
    return DischargeResult(sec.name, q, q_sd, area, q / area if area > 0 else float("nan"),
                           share, alpha, verts)


def continuity(a: DischargeResult, b: DischargeResult):
    """Relative disagreement between two sections, and whether it is within their combined uncertainty.

    α is shared by both sections, so it cancels out of the comparison; only the filled-vertical part counts.
    """
    diff = abs(a.q - b.q)
    mean = (a.q + b.q) / 2
    sd_a = 0.3 * a.q * (1 - a.measured_share)
    sd_b = 0.3 * b.q * (1 - b.measured_share)
    tol = 2 * np.hypot(sd_a, sd_b) + 0.10 * mean   # 10 % floor for sampling noise
    return {"rel_diff": float(round(diff / mean, 4)) if mean > 0 else None, "within": bool(diff <= tol),
            "tolerance_rel": float(round(tol / mean, 4)) if mean > 0 else None}


def run_lamorge(methods=("klt", "lspiv"), out=None):
    """Video -> Q at both surveyed La Morge sections, compared with the current-meter gauging."""
    import json
    from .cases import CASES, load_frames
    from .refs import lamorge_sections
    secs, ref = lamorge_sections()
    c = CASES["lamorge"]
    fr = load_frames(c)
    S = {k: Section(k, p, d, c.gsd) for k, (p, d) in secs.items()}
    rows = []
    for meth in methods:
        if meth == "klt":
            f, mn = M.kltiv_filter(M.klt(fr, c.fps, c.gsd)), 3
        else:
            f = M.lspiv(fr, c.fps, c.gsd, points=np.vstack([s.px for s in S.values()]), search=24)
            f, mn = f.subset(f.q > 0.15), 1
        rs = {k: discharge(f, s, min_n=mn) for k, s in S.items()}
        cont = continuity(rs["right"], rs["left"])
        for k, r in rs.items():
            row = dict(case="lamorge", method=meth, **r.as_dict(), q_ref=round(ref["q"], 4),
                       err=round(r.q / ref["q"] - 1, 4), continuity=cont)
            rows.append(row)
            print(f"{meth:6s} {k:6s} Q={r.q:.3f}±{r.q_sd:.3f} m³/s  gauged {ref['q']:.3f}  err {row['err']*100:+.1f}%  "
                  f"measured {r.measured_share*100:.0f}%  continuity {cont['rel_diff']*100:.0f}% "
                  f"({'pass' if cont['within'] else 'FAIL'})")
    if out:
        with open(out, "a") as fh:
            for r in rows:
                fh.write(json.dumps(r, default=float) + "\n")
    return rows


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    run_lamorge(out="results/discharge_lamorge.jsonl")
