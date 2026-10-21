"""Surface velocimetry methods. All OpenCV 5, one interface.

Each method takes a list of grey frames plus the case's fps and gsd, and returns a VectorField:
positions in pixels and velocities in m/s (image axes, +x right, +y down).

    klt    Shi-Tomasi features + pyramidal Lucas-Kanade, forward-backward check  (KLT-IV style)
    lspiv  interrogation-window cross-correlation, ensemble-averaged over all frame pairs
    stiv   space-time image along the flow direction, streak slope by total least squares
    dis    dense DIS optical flow, temporal median per pixel
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VectorField:
    pts: np.ndarray          # (N, 2) px
    vel: np.ndarray          # (N, 2) m/s
    q: np.ndarray            # (N,) per-vector quality in [0, 1]
    method: str
    meta: dict

    @property
    def speed(self):
        return np.linalg.norm(self.vel, axis=1)

    def subset(self, m):
        meta = dict(self.meta)
        if "pair" in meta and len(meta["pair"]) == len(self.pts):   # per-vector arrays follow the selection
            meta["pair"] = meta["pair"][m]
        return VectorField(self.pts[m], self.vel[m], self.q[m], self.method, meta)


def _pairs(n, gap):
    return [(i, i + gap) for i in range(0, n - gap)]


def flow_direction(field: VectorField) -> float:
    """Dominant flow direction (radians, image axes), speed-weighted circular mean."""
    s = field.speed
    a = np.arctan2(field.vel[:, 1], field.vel[:, 0])
    return float(np.arctan2(np.sum(s * np.sin(a)), np.sum(s * np.cos(a))))


def kltiv_filter(field: VectorField, min_speed=0.1, max_dev_deg=45.0) -> VectorField:
    """Filters from Perks et al. 2025 (KLT-IV): drop near-stationary vectors and those far off the flow line."""
    fast = field.speed >= min_speed
    if fast.sum() < 10:
        return field.subset(fast)
    d = flow_direction(field.subset(fast))
    a = np.arctan2(field.vel[:, 1], field.vel[:, 0])
    dev = np.abs(np.angle(np.exp(1j * (a - d))))
    return field.subset(fast & (dev < np.radians(max_dev_deg)))


# --------------------------------------------------------------------------- KLT
def klt(frames, fps, gsd, gap=1, win=31, levels=3, max_corners=4000, quality=0.005, fb_max=0.5, mask=None, **_):
    lk = dict(winSize=(win, win), maxLevel=levels,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    dt = gap / fps
    P, V, Q, K = [], [], [], []
    for i, j in _pairs(len(frames), gap):
        a, b = frames[i], frames[j]
        p0 = cv2.goodFeaturesToTrack(a, maxCorners=max_corners, qualityLevel=quality, minDistance=4, blockSize=7,
                                     mask=mask)   # inside the water only: banks can't crowd out the river
        if p0 is None:
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None, **lk)
        pb, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, p1, None, **lk)
        fb = np.linalg.norm(p0 - pb, axis=2).ravel()
        ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < fb_max)
        P.append(p0.reshape(-1, 2)[ok])
        V.append((p1 - p0).reshape(-1, 2)[ok] * gsd / dt)
        Q.append(1.0 - fb[ok] / fb_max)
        K.append(np.full(int(ok.sum()), i))
    if not P:
        return VectorField(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0), "klt", {"gap": gap, "pair": np.zeros(0, int)})
    return VectorField(np.vstack(P), np.vstack(V), np.concatenate(Q), "klt",
                       {"gap": gap, "win": win, "pair": np.concatenate(K)})


# --------------------------------------------------------------------------- LSPIV
def _subpix(c, i, j):
    """3-point Gaussian sub-pixel peak (falls back to parabolic when values are non-positive)."""
    def fit(m1, z, p1):
        if min(m1, z, p1) > 0:
            lm, lz, lp = np.log(m1), np.log(z), np.log(p1)
            den = 2 * lm - 4 * lz + 2 * lp
        else:
            lm, lz, lp = m1, z, p1
            den = 2 * lm - 4 * lz + 2 * lp
        return 0.0 if den == 0 else (lm - lp) / den
    h, w = c.shape
    dy = fit(c[i - 1, j], c[i, j], c[i + 1, j]) if 0 < i < h - 1 else 0.0
    dx = fit(c[i, j - 1], c[i, j], c[i, j + 1]) if 0 < j < w - 1 else 0.0
    return dx, dy


def lspiv(frames, fps, gsd, gap=1, ia=32, search=12, step=None, points=None, min_std=2.0, **_):
    """Ensemble-correlation LSPIV. `search` = max expected displacement (px per gap).
    points: optional (N,2) array of window centres; default = regular grid with `step` spacing."""
    h, w = frames[0].shape
    half = ia // 2
    step = step or ia // 2
    if points is None:
        ys, xs = np.mgrid[half + search: h - half - search: step, half + search: w - half - search: step]
        points = np.c_[xs.ravel(), ys.ravel()].astype(float)
    pts = np.round(points).astype(int)
    keep = (pts[:, 0] >= half + search) & (pts[:, 0] < w - half - search) & \
           (pts[:, 1] >= half + search) & (pts[:, 1] < h - half - search)
    pts = pts[keep]
    F = [f.astype(np.float32) for f in frames]
    # remove the static background (banks, glare that doesn't move) so correlation locks onto moving texture
    bg = np.median(np.stack(F[:: max(1, len(F) // 30)]), axis=0)
    F = [f - bg for f in F]
    acc = np.zeros((len(pts), 2 * search + 1, 2 * search + 1), np.float32)
    cnt = np.zeros(len(pts), np.int32)
    for i, j in _pairs(len(F), gap):
        a, b = F[i], F[j]
        for k, (x, y) in enumerate(pts):
            t = a[y - half: y + half, x - half: x + half]
            if t.std() < min_std:
                continue
            s = b[y - half - search: y + half + search, x - half - search: x + half + search]
            acc[k] += cv2.matchTemplate(s, t, cv2.TM_CCOEFF_NORMED)
            cnt[k] += 1
    vel = np.full((len(pts), 2), np.nan)
    q = np.zeros(len(pts))
    for k in range(len(pts)):
        if cnt[k] == 0:
            continue
        c = acc[k] / cnt[k]
        i, j = np.unravel_index(np.argmax(c), c.shape)
        dx, dy = _subpix(c, i, j)
        disp = np.array([j + dx - search, i + dy - search])
        vel[k] = disp * gsd * fps / gap
        # quality: peak height relative to the map's spread, and not on the search border
        border = i in (0, c.shape[0] - 1) or j in (0, c.shape[1] - 1)
        q[k] = 0.0 if border else float(np.clip((c.max() - c.mean()) / (c.std() * 6 + 1e-9), 0, 1))
    ok = np.isfinite(vel[:, 0])
    return VectorField(pts[ok].astype(float), vel[ok], q[ok], "lspiv",
                       {"gap": gap, "ia": ia, "search": search, "pairs": len(F) - gap})


# --------------------------------------------------------------------------- STIV
def _sti(F, x, y, u, length):
    """Space-time image (T x length) sampled along unit vector u through (x, y)."""
    T = F.shape[0]
    s = np.arange(length) - length / 2
    xs = (x + s * u[0]).astype(np.float32)[None, :]
    ys = (y + s * u[1]).astype(np.float32)[None, :]
    if xs.min() < 0 or ys.min() < 0 or xs.max() >= F.shape[2] - 1 or ys.max() >= F.shape[1] - 1:
        return None
    return np.stack([cv2.remap(F[t], xs, ys, cv2.INTER_LINEAR)[0] for t in range(T)])


def stiv_slope_fft(sti, vmax_px=None):
    """Streak slope (px/frame) from the 2-D power spectrum.
    A streak x = v*t + c puts its energy on the spectral line k_t = -v * k_x; we find the angle
    whose radial line carries the most energy. Returns (v_px_per_frame, coherence)."""
    T, L = sti.shape
    sti = sti - sti.mean(axis=1, keepdims=True)              # kill frame-wide flicker (energy at k_x = 0)
    sti = sti - sti.mean(axis=0, keepdims=True)              # and static texture (energy at k_t = 0)
    win = np.outer(np.hanning(T), np.hanning(L)).astype(np.float32)
    P = np.abs(np.fft.fftshift(np.fft.fft2((sti - sti.mean()) * win))) ** 2
    P = P.astype(np.float32)
    ct, cx = T // 2, L // 2
    # sample radial lines in normalised frequency units (cycles/sample on both axes)
    r = np.linspace(0.04, 0.45, 48)
    vmax = vmax_px or 12.0
    def energy(vs):
        out = []
        for v in vs:
            n = np.hypot(1, v)                               # unit direction (1, -v) / n in (k_x, k_t)
            mx = (cx + (r / n) * L).astype(np.float32)[None, :]
            mt = (ct + (-v * r / n) * T).astype(np.float32)[None, :]
            out.append(float(cv2.remap(P, mx, mt, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=0).sum()))
        return np.array(out)

    th = np.linspace(-np.arctan(vmax), np.arctan(vmax), 361)          # coarse: 361 angles
    e = energy(np.tan(th))
    k = int(np.argmax(e))
    coh = float((e[k] - np.median(e)) / (e[k] + 1e-9))
    step = th[1] - th[0]
    fine = np.linspace(th[k] - step, th[k] + step, 41)                 # fine: +-1 coarse step
    ef = energy(np.tan(fine))
    return float(np.tan(fine[int(np.argmax(ef))])), coh


def stiv(frames, fps, gsd, direction: float, points, length=96, gap=1, max_frames=128, vmax_px=12.0, **_):
    """Space-time image velocimetry along `direction` (radians) at each point (FFT streak angle)."""
    F = np.stack([f.astype(np.float32) for f in frames[::gap][:max_frames]])
    F -= F.mean(axis=0, keepdims=True)                      # remove the static background
    u = np.array([np.cos(direction), np.sin(direction)])
    out_v, out_q, out_p = [], [], []
    for (x, y) in np.asarray(points, float):
        sti = _sti(F, x, y, u, length)
        if sti is None:
            continue
        v_px, coh = stiv_slope_fft(sti, vmax_px)
        out_p.append((x, y)); out_v.append(v_px * gsd * fps / gap); out_q.append(max(coh, 0.0))
    v = np.array(out_v)
    vel = v[:, None] * u[None, :] if len(v) else np.zeros((0, 2))
    return VectorField(np.array(out_p).reshape(-1, 2), vel, np.array(out_q), "stiv",
                       {"direction": direction, "length": length, "gap": gap})


# --------------------------------------------------------------------------- DIS
def dis(frames, fps, gsd, gap=1, preset=cv2.DISOPTICAL_FLOW_PRESET_MEDIUM, step=4, max_pairs=60, max_width=960, **_):
    scale = min(1.0, max_width / frames[0].shape[1])          # full-res 4K DIS takes minutes; flow is smooth anyway
    if scale < 1.0:
        frames = [cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for f in frames]
        gsd = gsd / scale
    d = cv2.DISOpticalFlow_create(preset)
    pairs = _pairs(len(frames), gap)
    pairs = pairs[:: max(1, len(pairs) // max_pairs)]
    flows = np.stack([d.calc(frames[i], frames[j], None)[::step, ::step] for i, j in pairs])
    med = np.median(flows, axis=0)                           # (h, w, 2) px per gap
    mad = np.median(np.abs(flows - med), axis=0).sum(-1)
    h, w = med.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    pts = np.c_[xs.ravel(), ys.ravel()].astype(float) * step / scale
    vel = med.reshape(-1, 2) * gsd * fps / gap
    q = 1.0 / (1.0 + mad.ravel())
    return VectorField(pts, vel, q, "dis", {"gap": gap, "pairs": len(pairs)})


METHODS = {"klt": klt, "lspiv": lspiv, "stiv": stiv, "dis": dis}
