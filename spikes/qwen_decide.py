"""Can a Qwen model inside cv2.dnn choose the right repair? 6 scenarios x 2 prompt styles."""
import sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from gaugeless.agent.local import Qwen

q = Qwen(sys.argv[1])
LIM = {"moving_fraction": ("min", 0.02, "share of moving surface features"), "camera_motion_px": ("max", 1.0, "camera motion, pixels"),
       "repeated_frame_share": ("max", 0.3, "share of repeated frames"), "flicker": ("max", 0.05, "brightness flicker"),
       "glare_fraction": ("max", 0.05, "sun glare share")}
OPTS = [("stabilise", "Stabilise the camera (fixes camera motion)"), ("dedupe", "Remove repeated frames (fixes repeated frames)"),
        ("glare", "Mask sun glare (fixes glare)"), ("normalise", "Normalise brightness (fixes flicker)"),
        ("refuse", "Refuse (no moving surface features to track)"), ("measure", "Measure now (only if every check is ok)")]
base = {"moving_fraction": 0.45, "flicker": 0.002, "glare_fraction": 0.0, "camera_motion_px": 0.1, "repeated_frame_share": 0.0}
TESTS = {"clean": ({}, "measure"), "shake": ({"camera_motion_px": 6.9}, "stabilise"), "unseeded": ({"moving_fraction": 0.0}, "refuse"),
         "dupes": ({"repeated_frame_share": 0.5}, "dedupe"), "flicker": ({"flicker": 0.11}, "normalise"), "glare": ({"glare_fraction": 0.2}, "glare")}

def ask(c, labelled):
    if labelled:
        obs = "\n".join(f"- {d}: {c[k]} ({'PROBLEM' if (c[k] < l if t == 'min' else c[k] > l) else 'ok'})" for k, (t, l, d) in LIM.items())
    else:
        obs = "\n".join(f"- {d}: {c[k]} (limit: {'at least' if t == 'min' else 'at most'} {l})" for k, (t, l, d) in LIM.items())
    user = f"Checks on the video:\n{obs}\n\nActions:\n" + "\n".join(f"{'ABCDEF'[i]}. {d}" for i, (k, d) in enumerate(OPTS)) + \
           "\n\nWhich single action should come next? Answer with one letter."
    prompt = f"<|im_start|>system\nYou fix problems in river videos before measuring.<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    ids = np.array(q.tok.encode(prompt), np.int64).reshape(1, -1); n = ids.shape[1]
    q.net.resetKVCache(); q.net.setInput(ids, "input_ids"); q.net.setInput(np.ones((1, n), np.int64), "attention_mask")
    q.net.setInput(np.arange(n, dtype=np.int64).reshape(1, -1), "position_ids")
    lg = q.net.forward()[0, -1]; z = np.array([lg[t] for t in q.letter_ids[:6]]); p = np.exp(z - z.max()); p /= p.sum()
    return OPTS[int(np.argmax(p))][0], float(p.max())

for labelled in (False, True):
    ok = 0; t0 = time.time()
    for name, (d, want) in TESTS.items():
        got, p = ask(base | d, labelled); ok += got == want
        print(f"  {'labelled' if labelled else 'numbers '} {name:9s} want {want:9s} got {got:9s} p={p:.2f}")
    print(f"{'labelled' if labelled else 'numbers'}: {ok}/6 correct, {(time.time()-t0)/6:.2f}s per decision")
