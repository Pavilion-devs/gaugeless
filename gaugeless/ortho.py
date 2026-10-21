"""Orthorectification from ground control points (GCPs), for raw oblique or drone video.

    P = dlt(world_xyz, image_px)              # 3x4 camera matrix, 11-parameter DLT (no intrinsics needed)
    H = plane_homography(P, z_water)          # world (X, Y) on the water surface -> image pixel
    G = Grid(x0, y0, res, w, h)               # metric ortho grid, north up
    ortho = warp(frame, H, G)                 # cv2.warpPerspective with the inverse map

The water surface is the plane Z = stage (same datum as the GCPs), which is what makes a single homography
valid for every pixel of the river even though the GCPs themselves are at different heights.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def _normaliser(pts):
    c = pts.mean(axis=0)
    s = np.sqrt(pts.shape[1]) / np.mean(np.linalg.norm(pts - c, axis=1))
    T = np.eye(pts.shape[1] + 1)
    T[:-1, :-1] *= s
    T[:-1, -1] = -s * c
    return T


def dlt(world, px):
    """Direct linear transform, Hartley-normalised. world (N,3), px (N,2), N >= 6. Returns P (3x4), rms px."""
    world = np.asarray(world, float); px = np.asarray(px, float)
    Tw, Ti = _normaliser(world), _normaliser(px)
    W = (Tw @ np.c_[world, np.ones(len(world))].T).T
    I = (Ti @ np.c_[px, np.ones(len(px))].T).T
    A = []
    for (X, Y, Z, w), (u, v, _) in zip(W, I):
        A.append([X, Y, Z, w, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u * w])
        A.append([0, 0, 0, 0, X, Y, Z, w, -v * X, -v * Y, -v * Z, -v * w])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    Pn = Vt[-1].reshape(3, 4)
    P = np.linalg.inv(Ti) @ Pn @ Tw
    P /= np.linalg.norm(P[2, :3])
    proj = project(P, world)
    return P, float(np.sqrt(np.mean(np.sum((proj - px) ** 2, axis=1))))


def project(P, world):
    h = (P @ np.c_[world, np.ones(len(world))].T).T
    return h[:, :2] / h[:, 2:3]


def plane_homography(P, z):
    """Homography taking world (X, Y) on the plane Z = z to image pixels."""
    return np.c_[P[:, 0], P[:, 1], P[:, 2] * z + P[:, 3]]


@dataclass
class Grid:
    x0: float      # world X of the left edge, m
    y1: float      # world Y of the top edge, m (north up: rows go towards -Y)
    res: float     # m per px
    w: int
    h: int

    def to_world(self, uv):
        uv = np.asarray(uv, float)
        return np.c_[self.x0 + uv[:, 0] * self.res, self.y1 - uv[:, 1] * self.res]

    def matrix(self):
        """ortho pixel (u, v, 1) -> world (X, Y, 1)"""
        return np.array([[self.res, 0, self.x0], [0, -self.res, self.y1], [0, 0, 1]], float)


def warp(frame, H, grid: Grid, interp=cv2.INTER_LINEAR):
    """Resample an image frame onto the ortho grid. H maps world (X, Y) -> image px."""
    M = H @ grid.matrix()                       # ortho px -> image px
    return cv2.warpPerspective(frame, M, (grid.w, grid.h), flags=interp | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def grid_for(world_xy, res, margin=2.0):
    lo = world_xy.min(axis=0) - margin
    hi = world_xy.max(axis=0) + margin
    return Grid(float(lo[0]), float(hi[1]), res, int(np.ceil((hi[0] - lo[0]) / res)), int(np.ceil((hi[1] - lo[1]) / res)))


# ---------------------------------------------------------------- full camera model (lens distortion)

def calibrate(world, px, image_size, f_guesses=(900, 1200, 1600, 2000, 2600), k2=False):
    """Pinhole + radial distortion (k1, optionally k2) from one view of non-coplanar GCPs.
    Returns a camera dict with rms and leave-one-out reprojection error (px), the honest quality number."""
    W = np.asarray(world, np.float32); P = np.asarray(px, np.float32)
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST |
             cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_K3 | (0 if k2 else cv2.CALIB_FIX_K2))

    def fit(Wt, Pt, f0):
        K = np.array([[f0, 0, image_size[0] / 2], [0, f0, image_size[1] / 2], [0, 0, 1]], float)
        rms, K, d, rv, tv = cv2.calibrateCamera([Wt], [Pt], image_size, K, np.zeros(5), flags=flags)
        return rms, K, d, rv[0], tv[0]

    best = None
    for f0 in f_guesses:
        try:
            r = fit(W, P, f0)
        except cv2.error:
            continue
        if best is None or r[0] < best[0][0]:
            best = (r, f0)
    (rms, K, d, rv, tv), f0 = best
    loo = []
    for i in range(len(W)):
        m = np.arange(len(W)) != i
        try:
            _, Ki, di, rvi, tvi = fit(W[m], P[m], f0)
            p, _ = cv2.projectPoints(W[i:i + 1], rvi, tvi, Ki, di)
            loo.append(float(np.linalg.norm(p.ravel() - P[i])))
        except cv2.error:
            pass
    return {"K": K, "dist": d, "rvec": rv, "tvec": tv, "rms_px": float(rms),
            "loo_median_px": float(np.median(loo)), "loo_max_px": float(np.max(loo)), "image_size": tuple(image_size)}


def ortho_maps(cam, grid: Grid, z):
    """cv2.remap tables taking every ortho pixel (on the water plane Z = z) to its raw-image position,
    lens distortion included. Computed once per camera; each frame is then one cv2.remap."""
    u, v = np.meshgrid(np.arange(grid.w, dtype=np.float32), np.arange(grid.h, dtype=np.float32))
    world = np.stack([grid.x0 + u.ravel() * grid.res, grid.y1 - v.ravel() * grid.res,
                      np.full(u.size, z, np.float32)], axis=1).astype(np.float32)
    p, _ = cv2.projectPoints(world, cam["rvec"], cam["tvec"], cam["K"], cam["dist"])
    p = p.reshape(grid.h, grid.w, 2)
    R, _ = cv2.Rodrigues(cam["rvec"])
    zc = (world @ R[2].astype(np.float32) + float(cam["tvec"].ravel()[2])).reshape(grid.h, grid.w)
    p[zc <= 0] = np.nan                         # behind the camera: projectPoints would wrap these into view
    # points behind the camera or far outside the image map to nothing
    W, H = cam["image_size"]
    bad = ~np.isfinite(p).all(axis=2) | (p[..., 0] < -1) | (p[..., 0] > W) | (p[..., 1] < -1) | (p[..., 1] > H)
    p[bad] = -10
    return p[..., 0].astype(np.float32), p[..., 1].astype(np.float32)


def remap(frame, maps):
    return cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def register_to(ours, theirs, scale):
    """Similarity transform taking `theirs` pixels (full resolution) to `ours`, via SIFT + RANSAC. `scale` = nominal
    their m/px / our m/px. Published grids may be mirrored (Y up vs Y down), so all four flips are tried and the one
    with most RANSAC inliers wins. Used to carry reference positions given on a published grid onto ours."""
    t0 = cv2.resize(theirs, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, w = t0.shape[:2]
    sift = cv2.SIFT_create()
    k2, d2 = sift.detectAndCompute(ours, None)
    flips = {"none": np.eye(3), "flip_x": np.array([[-1, 0, w - 1], [0, 1, 0], [0, 0, 1]], float),
             "flip_y": np.array([[1, 0, 0], [0, -1, h - 1], [0, 0, 1]], float),
             "flip_xy": np.array([[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]], float)}
    best = None
    for name, F in flips.items():
        t = cv2.warpAffine(t0, F[:2], (w, h))
        k1, d1 = sift.detectAndCompute(t, None)
        m = cv2.BFMatcher().knnMatch(d1, d2, k=2)
        good = [a for a, b in m if a.distance < 0.8 * b.distance]
        if len(good) < 10:
            continue
        src = np.float32([k1[g.queryIdx].pt for g in good]); dst = np.float32([k2[g.trainIdx].pt for g in good])
        A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
        if A is None:
            continue
        n = int(inl.sum())
        if best is None or n > best[0]:
            resid = np.linalg.norm(cv2.transform(src[inl.ravel() == 1][None], A)[0] - dst[inl.ravel() == 1], axis=1)
            best = (n, name, F, A, len(good), float(np.median(resid)))
    n, name, F, A, ng, res = best
    T = np.vstack([A, [0, 0, 1]]) @ F @ np.diag([scale, scale, 1.0])     # theirs full-res -> ours
    return T[:2], {"flip": name, "matches": ng, "inliers": n, "resid_px": round(res, 2),
                   "scale_ratio": round(float(np.hypot(A[0, 0], A[1, 0])), 4),
                   "rotation_deg": round(float(np.degrees(np.arctan2(A[1, 0], A[0, 0]))), 2)}
