"""Controlled degradations of real clips, for the agent evaluation.

Each one reproduces a failure that field footage actually has. The clean clip's result is the truth for its
degraded versions, and the reference measurements are the truth for both.

    shake      hand-held / drone wobble: random-walk translation + rotation (px, deg)
    flicker    auto-exposure pumping: per-frame gain in [1-g, 1+g]
    blur       defocus / motion blur / rain on the lens: Gaussian sigma (px); destroys small tracers
    glare      sun glint: a saturated patch over a share of the frame + sparkles that move randomly
    duplicate  variable-frame-rate phone video re-encoded at a fixed rate: each frame repeated k times
"""
from __future__ import annotations

import cv2
import numpy as np


def shake(frames, amp=4.0, rot=0.3, seed=0):
    rng = np.random.default_rng(seed)
    h, w = frames[0].shape[:2]
    out, t, a = [], np.zeros(2), 0.0
    for i, f in enumerate(frames):
        if i:
            t = 0.7 * t + rng.normal(0, amp, 2)
            a = 0.7 * a + rng.normal(0, rot)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        M[:, 2] += t
        out.append(cv2.warpAffine(f, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
    return out


def flicker(frames, g=0.25, seed=0):
    rng = np.random.default_rng(seed)
    return [np.clip(f.astype(np.float32) * rng.uniform(1 - g, 1 + g), 0, 255).astype(np.uint8) for f in frames]


def blur(frames, sigma=3.0):
    return [cv2.GaussianBlur(f, (0, 0), sigma) for f in frames]


def glare(frames, frac=0.15, seed=0):
    rng = np.random.default_rng(seed)
    h, w = frames[0].shape[:2]
    patch = np.zeros((h, w), np.float32)
    ax = (int(w * np.sqrt(frac) * 0.6), int(h * np.sqrt(frac) * 0.6))
    cv2.ellipse(patch, (int(w * 0.5), int(h * 0.45)), ax, 20, 0, 360, 1.0, -1)
    patch = cv2.GaussianBlur(patch, (0, 0), max(ax) * 0.15 + 1)
    out = []
    for f in frames:
        g = f.astype(np.float32)
        g = g * (1 - patch) + 255 * patch
        n = int(h * w * 2e-4)
        ys, xs = rng.integers(0, h, n), rng.integers(0, w, n)
        for y, x in zip(ys, xs):
            if patch[y, x] > 0.2:
                cv2.circle(g, (int(x), int(y)), 2, 255, -1)
        out.append(np.clip(g, 0, 255).astype(np.uint8))
    return out


def duplicate(frames, k=2):
    return [f for f in frames for _ in range(k)]


DEGRADATIONS = {
    "shake": (shake, "amp", [2.0, 5.0, 10.0]),
    "flicker": (flicker, "g", [0.1, 0.25, 0.4]),
    "blur": (blur, "sigma", [1.5, 3.0, 6.0]),
    "glare": (glare, "frac", [0.05, 0.15, 0.3]),
    "duplicate": (duplicate, "k", [2, 3]),
}


def apply(frames, name, level):
    fn, arg, _ = DEGRADATIONS[name]
    return fn(frames, **{arg: level})
