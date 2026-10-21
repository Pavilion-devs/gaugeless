"""Every velocimetry method must recover a known synthetic translation (0.5-7 px/frame) within 3%."""
import cv2
import numpy as np
import pytest

from gaugeless import methods as M


def moving_texture(v, n=60, shape=(400, 600), angle=0.0):
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur(rng.random(shape).astype(np.float32), (0, 0), 2.0) * 255
    dx, dy = v * np.cos(angle), v * np.sin(angle)
    return [np.clip(cv2.warpAffine(base, np.float32([[1, 0, dx * t], [0, 1, dy * t]]), shape[::-1],
                                   borderMode=cv2.BORDER_REFLECT), 0, 255).astype(np.uint8) for t in range(n)]


PTS = np.array([[300, 200], [350, 150]])


@pytest.mark.parametrize("v", [0.5, 2.0, 4.0, 7.0])
def test_lspiv(v):
    f = M.lspiv(moving_texture(v), 1.0, 1.0, points=PTS, search=12)
    assert np.allclose(f.vel[:, 0], v, rtol=0.03, atol=0.03)


@pytest.mark.parametrize("v", [0.5, 2.0, 4.0, 7.0])
def test_stiv(v):
    f = M.stiv(moving_texture(v), 1.0, 1.0, direction=0.0, points=PTS)
    assert np.allclose(f.vel[:, 0], v, rtol=0.03, atol=0.03)


@pytest.mark.parametrize("v", [0.5, 2.0, 4.0, 7.0])
def test_klt(v):
    f = M.kltiv_filter(M.klt(moving_texture(v, n=15), 1.0, 1.0), min_speed=0.1)
    assert abs(np.median(f.vel[:, 0]) - v) < 0.03 * v + 0.03


def test_units_scale_with_gsd_and_fps():
    f = M.lspiv(moving_texture(4.0), fps=10.0, gsd=0.01, points=PTS, search=12)
    assert np.allclose(f.speed, 4.0 * 0.01 * 10.0, rtol=0.03)          # 0.4 m/s


def test_direction_is_recovered():
    f = M.kltiv_filter(M.klt(moving_texture(3.0, n=15, angle=np.radians(60)), 1.0, 1.0), min_speed=0.1)
    assert abs(np.degrees(M.flow_direction(f)) - 60) < 2
