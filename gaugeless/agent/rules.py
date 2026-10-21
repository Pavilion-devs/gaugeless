"""Deterministic planner: the ablation floor that the Claude and local-Qwen planners are compared against.

Each branch is tied to a measurement from the day-1 study (plan.md). The planner never sees reference values.
"""
from __future__ import annotations

import math

from . import tools as T

MAX_FRAMES_FIRST = 150
MAX_FRAMES_EXTENDED = 400
WANT_COVERAGE = 0.80


def plan(s: T.Session, call):
    """`call(name, **args)` runs a tool and returns its result (lets the runner log / intercept)."""
    clip = call("load_clip", max_frames=MAX_FRAMES_FIRST)
    if clip["repeated_frame_share"] > 0.3:
        call("remove_repeated_frames")
    d = call("diagnose_clip")
    if "camera_motion" in d["findings"]:
        st = call("stabilise")
        if ("error" in st or not st.get("kept")) and d["camera_motion_px"] > 3.0:
            call("refuse", reason=f"camera moves {d['camera_motion_px']:.1f} px and cannot be stabilised "
                                  f"({st.get('error') or st.get('reason')})",
                 advice="re-film from a tripod or a fixed mount with both banks in view")
            return
        if st.get("kept"):
            d = call("diagnose_clip")
    if "unsteady_bulk_motion" in d["findings"]:
        # shake the bank method could not see or remove (no bank in view, or bank not rigid)
        st = call("stabilise_from_flow")
        if not st.get("kept"):
            call("refuse", reason=f"camera shake: bulk motion jumps {d['flow_jitter']:.1f} px between frames and "
                                  f"could not be removed ({st.get('error') or st.get('reason')})",
                 advice="re-film from a tripod or a fixed mount, with a bank in view")
            return
        d = call("diagnose_clip")
    if "glare" in d["findings"]:
        call("mask_glare")
        d = call("diagnose_clip")

    if "no_trackable_surface" in d["findings"]:
        call("refuse", reason=f"no trackable surface (moving fraction {d['moving_fraction']:.4f}, "
                              f"tracer density {d['tracer_density']:.3f}/m²)",
             advice="seed the flow (e.g. biodegradable woodchips/cornstarch) or film when floating debris is present")
        return
    if "displacement_too_small" in d["findings"] and d["disp_px"] > 0:
        stride = min(10, max(2, math.ceil(2.0 / d["disp_px"])))
        call("load_clip", max_frames=MAX_FRAMES_FIRST, stride=stride)
        d = call("diagnose_clip")
    if "brightness_flicker" in d["findings"]:
        call("set_preprocess", mode="background")

    call("measure_velocity", method="klt")
    pts = None
    if s.targets is not None:
        pts = call("sample_points", method="klt")
        if pts["coverage"] < WANT_COVERAGE and clip["available"] > clip["frames"] * s.stride:
            clip = _reload(s, call, d)
            call("measure_velocity", method="klt")
            pts = call("sample_points", method="klt")
        if pts["coverage"] < WANT_COVERAGE:
            pts = call("sample_points", method="klt", fill_with="wide")

    if s.sections:
        q = call("compute_discharge", method="klt")
        if not q["passed"] and clip["available"] > clip["frames"] * s.stride:
            clip = _reload(s, call, d)
            call("measure_velocity", method="klt")
            q = call("compute_discharge", method="klt")
        if not q["passed"]:
            call("escalate", reason="; ".join(q["problems"]))
            return
        if s.flood_q and q["q"] >= s.flood_q:
            call("escalate", reason=f"Q {q['q']:.2f} m³/s at or above flood threshold {s.flood_q} m³/s")
            return
    elif pts is not None and pts["coverage"] < 0.5:
        call("escalate", reason=f"only {pts['coverage']:.0%} of gauge points measured")
        return
    call("publish")


def _reload(s, call, d):
    """Load the longer clip and repeat the repairs already chosen for this clip."""
    clip = call("load_clip", max_frames=MAX_FRAMES_EXTENDED, stride=s.stride)
    if clip["repeated_frame_share"] > 0.3:
        call("remove_repeated_frames")
    if any(t["tool"] == "stabilise" and t["result"].get("kept") for t in s.trace):
        call("stabilise")
    if any(t["tool"] == "stabilise_from_flow" and t["result"].get("kept") for t in s.trace):
        call("stabilise_from_flow")
    if any(t["tool"] == "mask_glare" for t in s.trace):
        call("mask_glare")
    return clip
