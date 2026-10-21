"""Surface diagnosis: measured properties of a clip that decide which method can be trusted.

Everything here is a number with units, not a label. The agent reads these to pick a method,
a frame stride, a region, or to refuse.

    tracer_density     moving trackable features per m² per frame (KLT survivors >= 0.1 m/s)
    moving_fraction    share of tracked features that move (vs static: banks, bed, glare)
    disp_px            median displacement of moving features per frame, px (target 2-8)
    dynamic_contrast   temporal std of the background-removed frames / static texture std
    glare_fraction     share of pixels saturated (>= 250) in at least half the frames
    flicker            frame-to-frame std of mean brightness / mean brightness
    direction_deg      dominant flow direction (image axes)
    direction_spread   circular std of moving-feature directions, degrees
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np

from . import methods as M


@dataclass
class Diagnosis:
    n_frames: int
    tracer_density: float
    moving_fraction: float
    disp_px: float
    dynamic_contrast: float
    glare_fraction: float
    flicker: float
    direction_deg: float
    direction_spread: float

    def as_dict(self):
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def diagnose(frames, fps, gsd, roi_mask=None, sample=30) -> Diagnosis:
    fr = frames[:sample]
    h, w = fr[0].shape
    area_m2 = (roi_mask.sum() if roi_mask is not None else h * w) * gsd * gsd
    raw = M.klt(fr, fps, gsd)
    if roi_mask is not None and len(raw.pts):
        inside = roi_mask[raw.pts[:, 1].astype(int).clip(0, h - 1), raw.pts[:, 0].astype(int).clip(0, w - 1)] > 0
        raw = raw.subset(inside)
    moving = M.kltiv_filter(raw)
    pairs = max(1, len(fr) - 1)
    density = len(moving.pts) / pairs / max(area_m2, 1e-9)
    moving_frac = len(moving.pts) / max(len(raw.pts), 1)
    disp = float(np.median(moving.speed) / gsd / fps) if len(moving.pts) else 0.0

    F = np.stack([f.astype(np.float32) for f in fr])
    bg = np.median(F, axis=0)
    dyn = float(np.std(F - bg) / (np.std(bg) + 1e-6))
    glare = float(np.mean(np.mean(F >= 250, axis=0) >= 0.5))
    means = F.mean(axis=(1, 2))
    flicker = float(np.std(np.diff(means)) / (means.mean() + 1e-6))

    if len(moving.pts) >= 10:
        d = M.flow_direction(moving)
        a = np.arctan2(moving.vel[:, 1], moving.vel[:, 0])
        R = np.abs(np.mean(np.exp(1j * a)))
        spread = float(np.degrees(np.sqrt(-2 * np.log(max(R, 1e-9)))))
    else:
        d, spread = float("nan"), float("nan")
    return Diagnosis(len(fr), float(density), float(moving_frac), disp, dyn, glare, flicker,
                     float(np.degrees(d)) if np.isfinite(d) else float("nan"), spread)


if __name__ == "__main__":
    import sys, warnings
    warnings.filterwarnings("ignore")
    from .cases import CASES, load_frames
    for n in sys.argv[1:]:
        c = CASES[n]
        print(f"{n:22s}", diagnose(load_frames(c, max_frames=30), c.fps, c.gsd).as_dict())
