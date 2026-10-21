"""Local planner: Qwen2.5-0.5B-Instruct running inside OpenCV 5's DNN engine (cv2.dnn + cv2.dnn.Tokenizer).
No PyTorch, no ONNX Runtime, no network: the edge / offline option.

A 0.5B model is not reliable at free-form multi-turn tool calling, so it makes *constrained choices*: at each
decision point it sees the measurements and a lettered list of the actions that are legal now, and the answer is
read from the next-token logits of the option letters (one forward pass, the answer is always well-formed).
The tools' guard rails still apply.

    GAUGELESS_QWEN=models/qwen2.5-0.5b python -m gaugeless.agent.run noce --planner local
"""
from __future__ import annotations

import json
import os
import time

import cv2
import numpy as np

from . import tools as T

MODEL_DIR = os.environ.get("GAUGELESS_QWEN", os.path.join(os.path.dirname(__file__), "..", "..", "models", "qwen2.5-0.5b"))
LETTERS = "ABCDEFG"

SYSTEM = ("You control a river velocity measurement from video. Choose the single best next action. "
          "Facts: if moving_fraction < 0.02 the surface has nothing to track and any number would be wrong, so refuse. "
          "If camera_motion_px > 1 the camera moves, so stabilise first. If repeated_frame_share > 0.3, remove repeated "
          "frames first. If flicker > 0.05, normalise brightness. If glare_fraction > 0.05, mask glare. "
          "Otherwise measure. Answer with one letter.")


class Qwen:
    def __init__(self, model_dir=MODEL_DIR):
        t0 = time.time()
        self.tok = cv2.dnn.Tokenizer.load(os.path.join(model_dir, "config.json"))
        self.net = cv2.dnn.readNetFromONNX(os.path.join(model_dir, "model.onnx"), cv2.dnn.ENGINE_NEW)
        self.net.enableKVCache()          # the export carries past_key_values; OpenCV routes present.* into them
        self.letter_ids = [self.tok.encode(L)[0] for L in LETTERS]
        self.load_s = time.time() - t0
        self.calls, self.tokens, self.seconds = 0, 0, 0.0

    def choose(self, context: dict, options: list[str]) -> tuple[int, list[float]]:
        opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
        user = f"Measurements: {json.dumps(context)}\nActions:\n{opts}\nAnswer with the letter only."
        prompt = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
                  f"<|im_start|>assistant\n")
        ids = np.array(self.tok.encode(prompt), np.int64).reshape(1, -1)
        n = ids.shape[1]
        t0 = time.time()
        self.net.resetKVCache()           # every decision is an independent prefill
        self.net.setInput(ids, "input_ids")
        self.net.setInput(np.ones((1, n), np.int64), "attention_mask")
        self.net.setInput(np.arange(n, dtype=np.int64).reshape(1, -1), "position_ids")
        logits = self.net.forward()[0, -1]
        self.seconds += time.time() - t0
        self.calls += 1
        self.tokens += n
        z = np.array([logits[t] for t in self.letter_ids[:len(options)]], float)
        p = np.exp(z - z.max()); p /= p.sum()
        return int(np.argmax(p)), p.round(3).tolist()


    def generate(self, system: str, user: str, max_new_tokens=120) -> str:
        """Greedy decoding with OpenCV's KV cache (prefill once, then one token per step)."""
        prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        ids = list(self.tok.encode(prompt))
        n = len(ids)
        t0 = time.time()
        self.net.resetKVCache()
        self.net.setInput(np.array(ids, np.int64).reshape(1, -1), "input_ids")
        self.net.setInput(np.ones((1, n), np.int64), "attention_mask")
        self.net.setInput(np.arange(n, dtype=np.int64).reshape(1, -1), "position_ids")
        nxt = int(np.argmax(self.net.forward()[0, -1]))
        out = [nxt]
        while len(out) < max_new_tokens and nxt not in (151645, 151643):
            cur = n + len(out)
            self.net.setInput(np.array([[nxt]], np.int64), "input_ids")
            self.net.setInput(np.ones((1, cur), np.int64), "attention_mask")
            self.net.setInput(np.array([[cur - 1]], np.int64), "position_ids")
            nxt = int(np.argmax(self.net.forward()[0, -1]))
            out.append(nxt)
        dt = time.time() - t0
        self.gen_stats = {"prompt_tokens": n, "new_tokens": len(out), "seconds": round(dt, 2),
                          "tokens_per_s": round(len(out) / dt, 1)}
        return self.tok.decode([t for t in out if t not in (151645, 151643)]).strip()


_QWEN = None


def report(s: T.Session) -> str:
    """Offline operator report written by Qwen inside OpenCV from the session's facts (decisions stay with the
    planner; the model only phrases them)."""
    global _QWEN
    if _QWEN is None:
        _QWEN = Qwen()
    facts = {"site": s.case.name, "camera": s.case.platform,
             "outcome": "published" if s.published else "refused" if s.refused else "waiting for a human" if s.escalated else "none",
             "repairs": [t["tool"] for t in s.trace if t["tool"] in ("stabilise", "remove_repeated_frames", "mask_glare", "set_preprocess")
                         and not t["result"].get("error") and t["result"].get("kept", True)]}
    if s.refused:
        facts["reason"] = s.refused
    if s.escalated:
        facts["reason"] = s.escalated
    if s.discharge:
        facts["discharge_m3s"] = s.discharge["q"]; facts["uncertainty_m3s"] = s.discharge["q_sd"]
        if "continuity" in s.discharge:
            facts["sections_agree_within_percent"] = round(100 * s.discharge["continuity"]["rel_diff"])
    if s.point_results is not None:
        v = s.point_results["v"]
        facts["points_measured"] = f"{int(np.isfinite(v).sum())} of {len(v)}"
        facts["median_surface_velocity_ms"] = round(float(np.nanmedian(v)), 2) if np.isfinite(v).any() else None
    sys_ = ("You write short field reports for hydrologists. Use only the facts given. Two or three plain sentences. "
            "State the outcome first, with the number and unit if there is one.")
    return _QWEN.generate(sys_, json.dumps(facts))


def plan(s: T.Session, call, verbose=True):
    global _QWEN
    if _QWEN is None:
        _QWEN = Qwen()
    q = _QWEN
    s.llm_usage = {"model": "Qwen2.5-0.5B-Instruct (cv2.dnn ENGINE_NEW)", "decisions": []}

    def decide(ctx, options, keys):
        i, p = q.choose(ctx, options)
        s.llm_usage["decisions"].append({"options": keys, "chosen": keys[i], "p": p})
        if verbose:
            print(f"  qwen: {dict(zip(keys, p))} -> {keys[i]}")
        return keys[i]

    clip = call("load_clip", max_frames=150)
    d = call("diagnose_clip")
    done = set()
    for _ in range(6):
        ctx = {k: d[k] for k in ("moving_fraction", "tracer_density", "disp_px", "flicker", "glare_fraction",
                                 "camera_motion_px")} | {"repeated_frame_share": clip["repeated_frame_share"]}
        menu = [("stabilise", "stabilise the camera using the banks"),
                ("dedupe", "remove repeated frames"),
                ("glare", "mask sun glare"),
                ("normalise", "normalise brightness flicker"),
                ("measure", "measure velocity now"),
                ("refuse", "refuse: nothing on the surface to track")]
        menu = [m for m in menu if m[0] not in done]
        a = decide(ctx, [m[1] for m in menu], [m[0] for m in menu])
        done.add(a)
        if a == "refuse":
            call("refuse", reason=f"local planner: no trackable surface (moving fraction {d['moving_fraction']})",
                 advice="seed the flow or film when floating debris is present")
            return
        if a == "measure":
            break
        if a == "stabilise":
            r = call("stabilise")
            if r.get("kept"):
                d = call("diagnose_clip")
        elif a == "dedupe":
            call("remove_repeated_frames"); d = call("diagnose_clip")
        elif a == "glare":
            call("mask_glare"); d = call("diagnose_clip")
        elif a == "normalise":
            call("set_preprocess", mode="background")

    r = call("measure_velocity", method="klt")
    if "refused" in r:                    # guard rail overrode the planner
        call("refuse", reason=r["refused"], advice="seed the flow or film when floating debris is present")
        return
    if s.targets is not None:
        pts = call("sample_points", method="klt")
        if pts["coverage"] < 0.8:
            a = decide({"coverage": pts["coverage"], "frames_loaded": clip["frames"], "frames_available": clip["available"]},
                       ["load more frames and measure again", "fill gaps from a wider radius", "accept as is"],
                       ["more", "wide", "accept"])
            if a == "more" and clip["available"] > clip["frames"]:
                clip = call("load_clip", max_frames=400)
                call("measure_velocity", method="klt"); pts = call("sample_points", method="klt")
            elif a == "wide":
                pts = call("sample_points", method="klt", fill_with="wide")
    if s.sections:
        res = call("compute_discharge", method="klt")
        if not res["passed"]:
            call("escalate", reason="; ".join(res["problems"]))
            return
        if s.flood_q and res["q"] >= s.flood_q:
            call("escalate", reason=f"Q {res['q']:.2f} m³/s at or above flood threshold {s.flood_q}")
            return
    call("publish")
    s.llm_usage.update({"calls": q.calls, "prompt_tokens": q.tokens, "seconds": round(q.seconds, 2),
                        "load_seconds": round(q.load_s, 1)})
