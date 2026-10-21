"""Agent tools: the only way a planner touches the pipeline.

Every tool reads and writes one `Session`, returns a JSON-able dict, and appends to the trace. Guard rails are
enforced here, not in the planner, so the rules planner, Claude on Bedrock and the local Qwen planner are held to
the same limits:

  * `measure_velocity` refuses when the diagnosis says there is nothing to track.
  * `compute_discharge` refuses without a surveyed section, below a coverage floor, or when the two-section
    continuity check fails beyond tolerance (unless a human has approved an override).
  * `publish` refuses until the result has passed checks or a human has approved it.

The tool specs in TOOL_SPECS are what the LLM planners see (MCP / Bedrock tool-use schema).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .. import methods as M
from .. import discharge as D
from ..diagnose import diagnose

# thresholds, from the day-1 measurements on 14 real clips (plan.md, "Day-1 findings")
MIN_MOVING_FRACTION = 0.02      # usable clips 0.21-0.98, unseeded 0.000-0.001
MIN_TRACER_DENSITY = 0.05       # usable clips >= 0.29 features/m²/frame
TARGET_DISP_PX = (1.0, 15.0)    # PyrLK with 3-5 pyramid levels is reliable in this range
FLICKER_PREP = 0.05             # above this, normalise brightness and remove the static background
POINT_IQR_FLAG = 0.30           # local relative IQR above this -> low-confidence point (error x2, spikes/point_conf.py)
COVERAGE_FLOOR = 0.60           # share of section discharge that must come from measured verticals
FROUDE_MAX = 1.5
FLOW_JITTER_MAX = 1.5           # px/frame; clean clips <= 0.84 on 11 sites, 2 px shake >= 2.5 (plan.md)


@dataclass
class Session:
    case: object                       # cases.Case (frames on disk, fps, gsd, water polygon)
    targets: np.ndarray | None = None  # (N, 2) px where velocities are wanted (gauge points)
    sections: dict | None = None       # name -> discharge.Section
    flood_q: float | None = None       # site alert threshold, m³/s
    frames: list = field(default_factory=list)
    frames_available: int = 0
    stride: int = 1
    time_factor: int = 1               # >1 after removing repeated frames: real time between kept frames
    fps_scale: float = 1.0             # evaluation only: file frame rate / real capture rate (re-encoded video)
    degrade: object = None             # evaluation only: callable applied to frames after loading
    preprocess: str = "none"
    mask: np.ndarray | None = None
    diagnosis: dict | None = None
    fields: dict = field(default_factory=dict)       # method -> VectorField
    point_results: dict | None = None
    discharge: dict | None = None
    approved: bool = False
    refused: str | None = None
    escalated: str | None = None
    published: dict | None = None
    trace: list = field(default_factory=list)

    def log(self, tool, args, result, t0):
        self.trace.append({"step": len(self.trace) + 1, "tool": tool, "args": args,
                           "result": result, "ms": int((time.time() - t0) * 1000)})
        return result


# ---------------------------------------------------------------- helpers

def _fps(s):
    return s.case.fps * s.fps_scale / (s.stride * s.time_factor)


def _bank_mask(mask):
    """Pixels that are certainly not water: outside the water polygon, eroded away from the edge."""
    if mask is None:
        return None
    k = max(5, int(0.03 * max(mask.shape)))
    bank = cv2.erode((mask == 0).astype(np.uint8), np.ones((k, k), np.uint8))
    return bank if bank.sum() > 0.02 * bank.size else None


def _bank_tracks(frames, mask, upto=None):
    bank = _bank_mask(mask)
    if bank is None:
        return None, None
    p0 = cv2.goodFeaturesToTrack(frames[0], 400, 0.01, 8, mask=bank * 255)
    if p0 is None or len(p0) < 20:
        return None, None
    return bank, p0


def _rigid_fits(frames, p0, idx):
    """Similarity transform frame_i -> frame_0 from bank features (RANSAC). Yields (i, A, inlier_share, resid_px)."""
    lk = dict(winSize=(31, 31), maxLevel=4)
    for i in idx:
        p1, st, _ = cv2.calcOpticalFlowPyrLK(frames[0], frames[i], p0, None, **lk)
        ok = st.ravel() == 1
        if ok.sum() < 10:
            yield i, None, 0.0, None; continue
        A, inl = cv2.estimateAffinePartial2D(p1[ok], p0[ok], method=cv2.RANSAC, ransacReprojThreshold=1.0)
        if A is None:
            yield i, None, 0.0, None; continue
        inl = inl.ravel().astype(bool)
        q = cv2.transform(p1[ok][inl], A)
        yield i, A, float(inl.mean()), float(np.median(np.linalg.norm((q - p0[ok][inl]).reshape(-1, 2), axis=1)))


def _corner_shift(A, shape):
    h, w = shape[:2]
    c = np.array([[[0, 0]], [[w, 0]], [[0, h]], [[w, h]]], np.float32)
    return float(np.max(np.linalg.norm((cv2.transform(c, A) - c).reshape(-1, 2), axis=1)))


def _camera_motion(frames, mask, n=12):
    """Camera motion = how far the rigid bank transform moves the frame corners, px (median over frames).
    Only counted when the bank behaves rigidly (>= 50 % RANSAC inliers); None when no bank is visible."""
    _, p0 = _bank_tracks(frames, mask)
    if p0 is None:
        return None
    sh = [_corner_shift(A, frames[0].shape) for _, A, share, _ in _rigid_fits(frames, p0, range(1, min(n, len(frames))))
          if A is not None and share >= 0.5]
    return round(float(np.median(sh)), 3) if sh else None


def _flow_jitter(frames, fps, gsd, mask, n=30):
    """Frame-to-frame instability of the bulk motion, px/frame: median over frame pairs of the distance between
    that pair's median displacement and the clip's median displacement. A river is steady over seconds; a
    shaking camera adds a different global shift to every pair. Needs no bank in view."""
    f = M.klt(frames[:n], fps, gsd, mask=mask)
    if len(f.pts) < 50:
        return None
    pair = f.meta["pair"]
    disp = f.vel / gsd / fps                    # px per frame
    overall = np.median(disp, axis=0)
    dev = [np.linalg.norm(np.median(disp[pair == k], axis=0) - overall) for k in np.unique(pair) if (pair == k).sum() >= 20]
    return round(float(np.median(dev)), 3) if dev else None


def _duplicates(frames):
    """Share of frames identical to their predecessor, and the typical repeat period."""
    if len(frames) < 3:
        return 0.0, 1
    diff = np.array([float(np.mean(cv2.absdiff(a, b))) for a, b in zip(frames[:-1], frames[1:])])
    dup = diff < 0.25
    runs, r = [], 1
    for x in dup:
        if x:
            r += 1
        else:
            runs.append(r); r = 1
    runs.append(r)
    period = int(np.median(runs)) if runs else 1
    return float(dup.mean()), max(1, period)

def _prep(frames, mode):
    if mode == "none":
        return frames
    fs = [f.astype(np.float32) for f in frames]
    m = float(np.mean([f.mean() for f in fs]))
    fs = [f * m / max(f.mean(), 1.0) for f in fs]                 # flicker: equalise frame brightness
    if mode == "normalise":
        return [np.clip(f, 0, 255).astype(np.uint8) for f in fs]
    bg = np.median(np.stack(fs), axis=0)                           # static texture: banks, bed, sensor pattern
    return [cv2.normalize(f - bg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) for f in fs]


def _local(field_, p, r):
    m = np.hypot(field_.pts[:, 0] - p[0], field_.pts[:, 1] - p[1]) <= r
    return field_.speed[m], m


# ---------------------------------------------------------------- tools

def load_clip(s: Session, max_frames: int = 150, stride: int = 1):
    """Load frames (frame QA drops odd-sized frames). More frames -> more tracer passes -> more coverage."""
    t0 = time.time()
    from ..cases import load_frames
    s.frames_available = s.case.available()
    s.frames = load_frames(s.case, max_frames=max_frames, stride=stride)
    if s.degrade is not None:
        s.frames = s.degrade(s.frames)
    s.stride = stride
    s.time_factor = 1
    dropped = getattr(load_frames, "last_dropped", 0)
    dup_share, period = _duplicates(s.frames)
    s.mask = s.case.water_mask(s.frames[0].shape) if s.frames else None
    s.fields.clear()
    res = {"frames": len(s.frames), "available": s.frames_available, "stride": stride,
           "effective_fps": round(_fps(s), 3), "dropped_bad_frames": dropped,
           "seconds_of_video": round(len(s.frames) * stride / s.case.fps, 1), "water_mask": s.mask is not None,
           "repeated_frame_share": round(dup_share, 3), "repeat_period": period}
    return s.log("load_clip", {"max_frames": max_frames, "stride": stride}, res, t0)


def diagnose_clip(s: Session):
    """Measured surface properties of the clip. Numbers with units, not labels."""
    t0 = time.time()
    fr = _prep(s.frames, s.preprocess)
    d = diagnose(fr, _fps(s), s.case.gsd, roi_mask=s.mask).as_dict()
    d["camera_motion_px"] = _camera_motion(s.frames, s.mask)
    d["flow_jitter"] = _flow_jitter(fr, _fps(s), s.case.gsd, s.mask)
    findings = []
    if d["moving_fraction"] < MIN_MOVING_FRACTION or d["tracer_density"] < MIN_TRACER_DENSITY:
        findings.append("no_trackable_surface")
    if d["disp_px"] and d["disp_px"] < TARGET_DISP_PX[0]:
        findings.append("displacement_too_small")
    if d["disp_px"] > TARGET_DISP_PX[1]:
        findings.append("displacement_too_large")
    if d["flicker"] > FLICKER_PREP:
        findings.append("brightness_flicker")
    if d["glare_fraction"] > 0.05:
        findings.append("glare")
    if d["camera_motion_px"] is not None and d["camera_motion_px"] > 1.0:
        findings.append("camera_motion")
    if d["flow_jitter"] is not None and d["flow_jitter"] > FLOW_JITTER_MAX:
        findings.append("unsteady_bulk_motion")
    if np.isfinite(d["direction_spread"]) and d["direction_spread"] > 30:
        findings.append("incoherent_directions")
    d["findings"] = findings
    s.diagnosis = d
    return s.log("diagnose_clip", {}, d, t0)


def set_preprocess(s: Session, mode: str):
    """'none' | 'normalise' (equalise brightness) | 'background' (normalise + remove static background)."""
    t0 = time.time()
    if mode not in ("none", "normalise", "background"):
        return s.log("set_preprocess", {"mode": mode}, {"error": "unknown mode"}, t0)
    s.preprocess = mode
    s.fields.clear()
    return s.log("set_preprocess", {"mode": mode}, {"preprocess": mode}, t0)


def remove_repeated_frames(s: Session):
    """Drop frames identical to their predecessor and correct the time step (a repeated frame means the
    file's frame rate is higher than the real capture rate, which silently divides every speed by the repeat)."""
    t0 = time.time()
    share, period = _duplicates(s.frames)
    keep = [s.frames[0]] + [b for a, b in zip(s.frames[:-1], s.frames[1:]) if float(np.mean(cv2.absdiff(a, b))) >= 0.25]
    s.frames = keep
    s.time_factor = period
    s.fields.clear()
    return s.log("remove_repeated_frames", {}, {"kept": len(keep), "repeat_period": period,
                                                "effective_fps": round(_fps(s), 3)}, t0)


def stabilise(s: Session):
    """Register every frame to frame 0 using bank features outside the water polygon (similarity, RANSAC).
    Self-checking: the result is kept only if camera motion drops and the bank fit is tight; otherwise the
    frames are restored and the tool says why (moving vegetation or water in the bank region)."""
    t0 = time.time()
    bank, p0 = _bank_tracks(s.frames, s.mask)
    if p0 is None:
        return s.log("stabilise", {}, {"error": "no stable bank features visible; re-film from a tripod"}, t0)
    before = _camera_motion(s.frames, s.mask)
    h, w = s.frames[0].shape[:2]
    n = len(s.frames)
    As = [np.array([[1, 0, 0], [0, 1, 0]], float)] + [None] * (n - 1)
    resid, shares = [], []
    for i, A, share, r in _rigid_fits(s.frames, p0, range(1, n)):
        if A is not None and share >= 0.3:
            As[i] = A; resid.append(r); shares.append(share)
    good = [i for i in range(n) if As[i] is not None]
    failed = n - len(good)
    # frames whose fit failed (flicker, blur): interpolate the camera motion from neighbours, never drop the
    # frame (dropping would double the time step of that pair and so double its speeds)
    for i in range(n):
        if As[i] is None:
            lo = max(j for j in good if j < i) if any(j < i for j in good) else None
            hi = min(j for j in good if j > i) if any(j > i for j in good) else None
            if lo is not None and hi is not None:
                t = (i - lo) / (hi - lo); As[i] = (1 - t) * As[lo] + t * As[hi]
            else:
                As[i] = As[lo if lo is not None else hi]
    out = [s.frames[0]] + [cv2.warpAffine(s.frames[i], As[i], (w, h), flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REPLICATE) for i in range(1, n)]
    res_px = float(np.median(resid)) if resid else None
    old = s.frames
    s.frames = out
    after = _camera_motion(s.frames, s.mask)
    ok = res_px is not None and res_px <= 1.0 and failed <= 0.3 * len(old) and \
        (before is None or after is None or after < before)
    res = {"bank_features": int(len(p0)), "interpolated_frames": failed, "inlier_share": round(float(np.median(shares)), 3) if shares else None,
           "residual_px": None if res_px is None else round(res_px, 3),
           "camera_motion_before_px": before, "camera_motion_after_px": after, "kept": ok}
    if not ok:
        s.frames = old
        res["reason"] = "bank does not move rigidly (vegetation, water or parallax in the bank region); frames restored"
    s.fields.clear()
    return s.log("stabilise", {}, res, t0)


def _pair_shifts(frames, fps, gsd, mask, cell=32):
    """Per-pair camera shift, px: median over vectors of (displacement - time-median displacement in that vector's
    grid cell). Comparing each vector with its own location's steady motion makes the estimate independent of
    which parts of the river happened to have trackable features in that frame."""
    f = M.klt(frames, fps, gsd, mask=mask)
    if len(f.pts) < 50:
        return None
    pair = f.meta["pair"]
    disp = f.vel / gsd / fps
    h, w = frames[0].shape[:2]
    cid = (f.pts[:, 1] // cell).astype(int) * (w // cell + 1) + (f.pts[:, 0] // cell).astype(int)
    order = np.argsort(cid)
    uniq, start = np.unique(cid[order], return_index=True)
    ref = np.zeros_like(disp)
    for u, s0, s1 in zip(uniq, start, np.r_[start[1:], len(order)]):
        idx = order[s0:s1]
        ref[idx] = np.median(disp[idx], axis=0)
    resid = disp - ref
    steps = np.zeros((len(frames) - 1, 2))
    for k in range(len(frames) - 1):
        m = pair == k
        if m.sum() >= 20:
            steps[k] = np.median(resid[m], axis=0)
    return steps


def stabilise_from_flow(s: Session):
    """Shake compensation when no bank is visible. Each frame pair's shift is its motion relative to the
    time-median motion at the same locations (rivers are steady over seconds); shifts are accumulated and removed
    by translation, in two passes. Self-checking: kept only if bulk-motion jitter falls below the limit."""
    t0 = time.time()
    fps, gsd = _fps(s), s.case.gsd
    before = _flow_jitter(s.frames, fps, gsd, s.mask)
    old = s.frames
    h, w = old[0].shape[:2]
    total = np.zeros((len(old), 2))
    for _ in range(2):
        steps = _pair_shifts(s.frames, fps, gsd, s.mask)
        if steps is None:
            s.frames = old
            return s.log("stabilise_from_flow", {}, {"error": "too few features to estimate shake"}, t0)
        total += np.vstack([np.zeros(2), np.cumsum(steps, axis=0)])
        s.frames = [cv2.warpAffine(fr, np.float32([[1, 0, -dx], [0, 1, -dy]]), (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE) for fr, (dx, dy) in zip(old, total)]
    after = _flow_jitter(s.frames, fps, gsd, s.mask)
    ok = after is not None and after <= FLOW_JITTER_MAX and (before is None or after < before)
    res = {"jitter_before_px": before, "jitter_after_px": after, "max_shift_px": round(float(np.abs(total).max()), 1), "kept": ok}
    if not ok:
        s.frames = old
        res["reason"] = "bulk motion still unsteady after compensation; frames restored"
    s.fields.clear()
    return s.log("stabilise_from_flow", {}, res, t0)


def mask_glare(s: Session):
    """Remove pixels saturated in >= 30 % of frames (sun glint) from the measurement region."""
    t0 = time.time()
    F = np.stack(s.frames[:60])
    g = (np.mean(F >= 245, axis=0) >= 0.3).astype(np.uint8)
    k = max(5, int(0.01 * max(g.shape)) | 1)
    g = cv2.dilate(g, np.ones((k, k), np.uint8))
    base = s.mask if s.mask is not None else np.ones(g.shape, np.uint8)
    before = int(base.sum())
    s.mask = (base * (1 - g)).astype(np.uint8)
    s.fields.clear()
    return s.log("mask_glare", {}, {"removed_share_of_region": round(1 - s.mask.sum() / max(before, 1), 3)}, t0)


def measure_velocity(s: Session, method: str = "klt", **params):
    """Run one velocimetry method on the loaded clip. Guard rail: refuses on an untrackable surface."""
    t0 = time.time()
    args = {"method": method, **params}
    if s.diagnosis and "no_trackable_surface" in s.diagnosis["findings"]:
        return s.log("measure_velocity", args, {"refused": "diagnosis found no trackable surface; any velocity "
                                                "from this clip would be a silent failure (Arrow unseeded: -95 to -100%)"}, t0)
    fr = _prep(s.frames, s.preprocess)
    fps = _fps(s)
    gsd = s.case.gsd
    if method == "klt":
        f = M.kltiv_filter(M.klt(fr, fps, gsd, mask=s.mask, **params))
    elif method == "lspiv":
        pts = _wanted_points(s)
        f = M.lspiv(fr, fps, gsd, points=pts, search=params.pop("search", 24), **params)
        f = f.subset(f.q > 0.15)
    elif method == "stiv":
        d = np.radians(s.diagnosis["direction_deg"]) if s.diagnosis and np.isfinite(s.diagnosis["direction_deg"]) else 0.0
        f = M.stiv(fr, fps, gsd, direction=d, points=_wanted_points(s), **params)
    else:
        return s.log("measure_velocity", args, {"error": f"unknown method {method}"}, t0)
    s.fields[method] = f
    res = {"method": method, "vectors": int(len(f.pts)),
           "median_speed_ms": round(float(np.median(f.speed)), 3) if len(f.pts) else None}
    return s.log("measure_velocity", args, res, t0)


def _wanted_points(s: Session):
    pts = []
    if s.targets is not None:
        pts.append(np.asarray(s.targets, float))
    if s.sections:
        pts += [sec.px for sec in s.sections.values()]
    return np.vstack(pts) if pts else None


def sample_points(s: Session, method: str = "klt", radius_m: float = 0.3, fill_with: str | None = None):
    """Velocity at each target point, with a confidence flag from local agreement.

    fill_with='wide': points with no vectors take the same method's median from a 2.5x wider circle
    (low-confidence). fill_with='lspiv': gaps take the LSPIV value, only if LSPIV agrees with KLT on the points
    both measured (median |ratio-1| < 20 %). On real rivers 'wide' was more accurate than LSPIV filling.
    """
    t0 = time.time()
    args = {"method": method, "radius_m": radius_m, "fill_with": fill_with}
    f = s.fields.get(method)
    if f is None or s.targets is None:
        return s.log("sample_points", args, {"error": "measure first / no targets"}, t0)
    r = float(np.clip(radius_m / s.case.gsd, 8, 60))
    v = np.full(len(s.targets), np.nan)
    iqr = np.full(len(s.targets), np.nan)
    src = np.array([""] * len(s.targets), dtype=object)
    for i, p in enumerate(s.targets):
        sp, _ = _local(f, p, r)
        if len(sp) >= 3:
            med = np.median(sp)
            v[i] = med
            iqr[i] = (np.percentile(sp, 75) - np.percentile(sp, 25)) / max(med, 1e-6)
            src[i] = method
    fill_note = None
    if fill_with == "wide":
        # gaps take the same method's vectors from a 2.5x wider circle (spikes/actions.py: +15-25 points of
        # coverage for 0-3 points of MAPE); always flagged low-confidence
        r2 = float(np.clip(2.5 * radius_m / s.case.gsd, 20, 150))
        gap = ~np.isfinite(v)
        for i in np.flatnonzero(gap):
            sp, _ = _local(f, s.targets[i], r2)
            if len(sp) >= 3:
                v[i] = np.median(sp); src[i] = "wide"
        filled = int((src == "wide").sum())
        fill_note = {"filled": filled, "radius_m": round(r2 * s.case.gsd, 2)}
    elif fill_with and fill_with in s.fields:
        g = s.fields[fill_with]
        gv = np.full(len(s.targets), np.nan)
        for i, p in enumerate(s.targets):
            sp, _ = _local(g, p, max(r, 8))
            if len(sp) >= 1:
                gv[i] = np.median(sp)
        both = np.isfinite(v) & np.isfinite(gv) & (v > 0.05)
        agree = float(np.median(np.abs(gv[both] / v[both] - 1))) if both.sum() >= 3 else None
        if agree is not None and agree < 0.20:
            gap = ~np.isfinite(v) & np.isfinite(gv)
            v[gap] = gv[gap]; src[gap] = fill_with
            fill_note = {"agreement_on_overlap": round(agree, 3), "filled": int(gap.sum())}
        else:
            fill_note = {"agreement_on_overlap": None if agree is None else round(agree, 3), "filled": 0,
                         "reason": "fill method disagrees with primary on overlap" if agree is not None else "too little overlap"}
    low_conf = (np.isfinite(iqr) & (iqr > POINT_IQR_FLAG)) | (src == "wide") | (src == "lspiv")
    s.point_results = {"v": v, "iqr": iqr, "source": src, "low_conf": low_conf}
    res = {"points": len(v), "measured": int(np.isfinite(v).sum()), "coverage": round(float(np.isfinite(v).mean()), 3),
           "low_confidence": int(low_conf.sum()), "median_ms": round(float(np.nanmedian(v)), 3) if np.isfinite(v).any() else None,
           "fill": fill_note}
    return s.log("sample_points", args, res, t0)


def compute_discharge(s: Session, method: str = "klt", alpha: float = D.ALPHA):
    """Q at every surveyed section + continuity between them. Guard rails inside."""
    t0 = time.time()
    args = {"method": method, "alpha": alpha}
    if not s.sections:
        return s.log("compute_discharge", args, {"refused": "no surveyed cross-section for this site"}, t0)
    f = s.fields.get(method)
    if f is None:
        return s.log("compute_discharge", args, {"error": "measure first"}, t0)
    rs = {k: D.discharge(f, sec, alpha=alpha, min_n=3 if method == "klt" else 1) for k, sec in s.sections.items()}
    out = {"method": method, "sections": {k: r.as_dict() for k, r in rs.items()}}
    problems = []
    for k, r in rs.items():
        if r.measured_share < COVERAGE_FLOOR:
            problems.append(f"{k}: only {r.measured_share:.0%} of Q from measured verticals")
        depth = r.area / max(s.sections[k].station[-1], 1e-6)
        fr_ = (r.mean_velocity / alpha) / np.sqrt(9.81 * max(depth, 1e-3))
        out["sections"][k]["froude"] = round(float(fr_), 3)
        if fr_ > FROUDE_MAX:
            problems.append(f"{k}: Froude {fr_:.2f} implausible")
    if len(rs) >= 2:
        a, b = list(rs.values())[:2]
        cont = D.continuity(a, b)
        out["continuity"] = cont
        if not cont["within"]:
            problems.append(f"continuity: sections disagree by {cont['rel_diff']:.0%} (tolerance {cont['tolerance_rel']:.0%})")
    qs = [r.q for r in rs.values()]
    out["q"] = round(float(np.mean(qs)), 4)
    out["q_sd"] = round(float(np.sqrt(np.mean([r.q_sd ** 2 for r in rs.values()]) + np.var(qs))), 4)
    out["problems"] = problems
    out["passed"] = not problems
    s.discharge = out
    return s.log("compute_discharge", args, out, t0)


def escalate(s: Session, reason: str):
    """Hand the decision to a human (in AWS: Step Functions waits on a task token; locally: the trace says so)."""
    t0 = time.time()
    s.escalated = reason
    return s.log("escalate", {"reason": reason}, {"status": "awaiting_human", "reason": reason}, t0)


def refuse(s: Session, reason: str, advice: str):
    t0 = time.time()
    s.refused = reason
    return s.log("refuse", {"reason": reason, "advice": advice}, {"status": "refused", "reason": reason, "advice": advice}, t0)


def publish(s: Session):
    """Release the result. Guard rail: only checked results, or ones a human approved."""
    t0 = time.time()
    if s.refused:
        return s.log("publish", {}, {"refused": "session was refused: " + s.refused}, t0)
    if s.targets is not None and s.point_results is None:
        return s.log("publish", {}, {"refused": "this site has gauge points: run sample_points before publishing"}, t0)
    if s.sections and s.discharge is None:
        return s.log("publish", {}, {"refused": "this site has surveyed sections: run compute_discharge before publishing"}, t0)
    if s.discharge is not None and not s.discharge["passed"] and not s.approved:
        return s.log("publish", {}, {"refused": "discharge failed checks and has no human approval",
                                     "problems": s.discharge["problems"]}, t0)
    flood = bool(s.flood_q and s.discharge and s.discharge["q"] >= s.flood_q)
    if flood and not s.approved:
        return s.log("publish", {}, {"refused": "flood threshold crossed: alert needs human approval"}, t0)
    out = {"status": "published", "flood_alert": flood}
    if s.point_results is not None:
        pr = s.point_results
        out["points"] = [None if not np.isfinite(x) else round(float(x), 3) for x in pr["v"]]
        out["low_confidence"] = [bool(x) for x in pr["low_conf"]]
    if s.discharge is not None:
        out["q"], out["q_sd"] = s.discharge["q"], s.discharge["q_sd"]
    s.published = out
    return s.log("publish", {}, {k: v for k, v in out.items() if k not in ("points", "low_confidence")}, t0)


TOOLS = {f.__name__: f for f in [load_clip, diagnose_clip, remove_repeated_frames, stabilise, stabilise_from_flow, mask_glare,
                                 set_preprocess, measure_velocity, sample_points,
                                 compute_discharge, escalate, refuse, publish]}

TOOL_SPECS = [
    {"name": "load_clip", "description": load_clip.__doc__.strip(),
     "input_schema": {"type": "object", "properties": {"max_frames": {"type": "integer", "minimum": 10, "maximum": 600},
                                                       "stride": {"type": "integer", "minimum": 1, "maximum": 10}}}},
    {"name": "diagnose_clip", "description": diagnose_clip.__doc__.strip(), "input_schema": {"type": "object", "properties": {}}},
    {"name": "remove_repeated_frames", "description": remove_repeated_frames.__doc__.strip(),
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "stabilise", "description": stabilise.__doc__.strip(), "input_schema": {"type": "object", "properties": {}}},
    {"name": "stabilise_from_flow", "description": stabilise_from_flow.__doc__.strip(), "input_schema": {"type": "object", "properties": {}}},
    {"name": "mask_glare", "description": mask_glare.__doc__.strip(), "input_schema": {"type": "object", "properties": {}}},
    {"name": "set_preprocess", "description": set_preprocess.__doc__.strip(),
     "input_schema": {"type": "object", "properties": {"mode": {"type": "string", "enum": ["none", "normalise", "background"]}},
                      "required": ["mode"]}},
    {"name": "measure_velocity", "description": measure_velocity.__doc__.strip(),
     "input_schema": {"type": "object", "properties": {"method": {"type": "string", "enum": ["klt", "lspiv", "stiv"]}},
                      "required": ["method"]}},
    {"name": "sample_points", "description": sample_points.__doc__.strip(),
     "input_schema": {"type": "object", "properties": {"method": {"type": "string"}, "radius_m": {"type": "number"},
                                                       "fill_with": {"type": "string", "enum": ["wide", "lspiv", "stiv"]}}}},
    {"name": "compute_discharge", "description": compute_discharge.__doc__.strip(),
     "input_schema": {"type": "object", "properties": {"method": {"type": "string"}, "alpha": {"type": "number"}}}},
    {"name": "escalate", "description": escalate.__doc__.strip(),
     "input_schema": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}},
    {"name": "refuse", "description": "Stop: this clip cannot give a trustworthy number. Say why and what the operator should do.",
     "input_schema": {"type": "object", "properties": {"reason": {"type": "string"}, "advice": {"type": "string"}},
                      "required": ["reason", "advice"]}},
    {"name": "publish", "description": publish.__doc__.strip(), "input_schema": {"type": "object", "properties": {}}},
]
