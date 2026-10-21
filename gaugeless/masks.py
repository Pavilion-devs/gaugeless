"""Water-region masks.

Operational practice (e.g. KLT-IV) is for the operator to outline the water once per site. We support
that (polygon in the site config) and also propose a mask automatically, which the agent can accept,
reject, or replace:

  candidate "dynamic": pixels whose temporal variation is high relative to their static texture
  candidate "smooth":  pixels with little static texture (dark / calm water next to textured banks)

The candidate whose tracked motion is fastest and most coherent in direction wins, because that is
what flowing water does and banks / bed / vegetation don't.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import methods as M


def _clean(mask, k):
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3 * k, 3 * k), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return mask
    return (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)


def _motion_score(field: M.VectorField, mask):
    """Median speed x direction coherence of moving vectors inside the mask."""
    if len(field.pts) == 0:
        return 0.0, 0
    h, w = mask.shape
    inside = mask[field.pts[:, 1].astype(int).clip(0, h - 1), field.pts[:, 0].astype(int).clip(0, w - 1)] > 0
    f = M.kltiv_filter(field.subset(inside))
    if len(f.pts) < 20:
        return 0.0, len(f.pts)
    a = np.arctan2(f.vel[:, 1], f.vel[:, 0])
    coherence = float(np.abs(np.mean(np.exp(1j * a))))
    return float(np.median(f.speed) * coherence), len(f.pts)


def auto_water_mask(frames, fps, gsd, n=40):
    fr = frames[:n]
    g = np.stack([f.astype(np.float32) for f in fr])
    bg = np.median(g, axis=0)
    k = max(9, int(0.15 / gsd) | 1)
    mu = cv2.blur(bg, (k, k))
    tex = np.sqrt(np.maximum(cv2.blur(bg * bg, (k, k)) - mu * mu, 0))
    dyn = cv2.blur(np.std(g - bg, axis=0), (k, k))
    ratio = dyn / (tex + 1)
    cands = {
        "dynamic": _clean(ratio > np.percentile(ratio, 50), k),
        "smooth": _clean(tex < np.percentile(tex, 40), k),
    }
    field = M.klt(fr[: min(len(fr), 20)], fps, gsd)
    scores = {name: _motion_score(field, m) for name, m in cands.items()}
    best = max(scores, key=lambda s: scores[s][0])
    return cands[best], {"chosen": best, "scores": {k_: {"score": round(v[0], 4), "vectors": v[1]}
                                                     for k_, v in scores.items()},
                         "water_fraction": float(cands[best].mean())}
