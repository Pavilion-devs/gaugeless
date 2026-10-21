"""Guard rails and repair tools on synthetic clips (no dataset needed)."""
import numpy as np
import pytest

from gaugeless import degrade as G
from gaugeless.agent import tools as T
from gaugeless.discharge import Section


class FakeCase:
    """Textured banks top and bottom, a moving textured river in the middle band."""
    name, platform, country, ref = "synthetic", "test", "XX", None

    def __init__(self, n=40, speed_px=3.0, h=240, w=320, seeded=True, fps=10.0, gsd=0.01):
        rng = np.random.default_rng(1)
        self.fps, self.gsd = fps, gsd
        big = (rng.random((h, w + n * int(speed_px) + 8)) * 255).astype(np.uint8)
        bank = (rng.random((h, w)) * 255).astype(np.uint8)
        frames = []
        for i in range(n):
            f = bank.copy()
            off = int(round(i * speed_px))
            river = big[:, off:off + w] if seeded else np.full((h, w), 120, np.uint8)
            f[80:160] = river[80:160]
            frames.append(f)
        self._frames = frames
        self.water = ((0, 80), (w, 80), (w, 160), (0, 160))

    def available(self):
        return len(self._frames)

    def water_mask(self, shape):
        m = np.zeros(shape[:2], np.uint8); m[80:160] = 1
        return m


@pytest.fixture
def session(monkeypatch):
    def make(**kw):
        c = FakeCase(**kw)
        import gaugeless.cases as C
        monkeypatch.setattr(C, "load_frames", lambda case, max_frames=None, stride=1, color=False:
                            case._frames[::stride][:max_frames])
        return T.Session(case=c, targets=np.array([[100.0, 120.0], [200.0, 120.0]]))
    return make


def test_measures_known_speed(session):
    s = session()
    T.load_clip(s); T.diagnose_clip(s); T.measure_velocity(s, "klt"); r = T.sample_points(s)
    assert r["coverage"] == 1.0
    assert r["median_ms"] == pytest.approx(3.0 * 0.01 * 10, rel=0.05)     # 0.3 m/s


def test_refuses_untrackable_surface(session):
    s = session(seeded=False)
    T.load_clip(s); d = T.diagnose_clip(s)
    assert "no_trackable_surface" in d["findings"]
    assert "refused" in T.measure_velocity(s, "klt")


def test_repeated_frames_detected_and_time_corrected(session):
    s = session()
    s.degrade = lambda fr: G.duplicate(fr, 3)
    s.fps_scale = 3                       # file claims 30 fps, camera captured 10
    c = T.load_clip(s)
    assert c["repeated_frame_share"] > 0.6 and c["repeat_period"] == 3
    T.remove_repeated_frames(s); T.diagnose_clip(s); T.measure_velocity(s, "klt")
    assert T.sample_points(s)["median_ms"] == pytest.approx(0.3, rel=0.05)


def test_stabilise_removes_shake(session):
    s = session()
    s.degrade = lambda fr: G.shake(fr, amp=4.0, rot=0.0)
    T.load_clip(s); d = T.diagnose_clip(s)
    assert "camera_motion" in d["findings"]
    r = T.stabilise(s)
    assert r["kept"] and r["camera_motion_after_px"] < 0.5
    T.measure_velocity(s, "klt")
    assert T.sample_points(s)["median_ms"] == pytest.approx(0.3, rel=0.08)


def test_publish_requires_points_sampled(session):
    s = session()
    T.load_clip(s); T.diagnose_clip(s); T.measure_velocity(s, "klt")
    assert "refused" in T.publish(s)
    T.sample_points(s)
    assert T.publish(s)["status"] == "published"


def test_publish_blocked_without_approval_when_checks_fail(session):
    s = session()
    s.point_results = {"v": np.array([0.3, 0.3]), "low_conf": np.array([False, False])}
    s.discharge = {"passed": False, "problems": ["continuity"], "q": 1.0, "q_sd": 0.1}
    assert "refused" in T.publish(s)
    s.approved = True
    assert T.publish(s)["status"] == "published"


def test_flood_alert_needs_human(session):
    s = session()
    s.point_results = {"v": np.array([0.3, 0.3]), "low_conf": np.array([False, False])}
    s.flood_q = 0.5
    s.discharge = {"passed": True, "problems": [], "q": 1.0, "q_sd": 0.1}
    assert "refused" in T.publish(s)


def test_discharge_needs_section(session):
    s = session()
    T.load_clip(s); T.measure_velocity(s, "klt")
    assert "refused" in T.compute_discharge(s)


def test_discharge_on_synthetic_channel(session):
    s = session()
    # one section across the river band, 1 m deep everywhere, 0.8 m wide (80 px)
    px = np.stack([np.full(9, 160.0), np.linspace(80, 160, 9)], axis=1)
    s.sections = {"a": Section("a", px, np.ones(9), 0.01)}
    T.load_clip(s); T.diagnose_clip(s); T.measure_velocity(s, "klt")
    q = T.compute_discharge(s, alpha=1.0)
    assert q["q"] == pytest.approx(0.3 * 0.8 * 1.0, rel=0.08)


def test_rules_planner_repairs_and_publishes(session):
    from gaugeless.agent import rules
    s = session()
    s.degrade = lambda fr: G.duplicate(G.shake(fr, amp=3.0, rot=0.0), 2)
    s.fps_scale = 2
    rules.plan(s, lambda t, **a: T.TOOLS[t](s, **a))
    tools = [t["tool"] for t in s.trace]
    assert "remove_repeated_frames" in tools and "stabilise" in tools
    assert s.published is not None
    assert np.nanmedian(s.published["points"]) == pytest.approx(0.3, rel=0.08)


def test_rules_planner_refuses_unseeded(session):
    from gaugeless.agent import rules
    s = session(seeded=False)
    rules.plan(s, lambda t, **a: T.TOOLS[t](s, **a))
    assert s.refused and s.published is None
