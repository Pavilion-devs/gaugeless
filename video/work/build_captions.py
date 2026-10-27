"""Caption groups from the voiceover word timings. Re-run after changing a VO line or its start time in index.html."""
import json, re
OFF = {"hook": 0.3, "problem": 12.2, "intro": 27.4, "shake": 39.6, "refuse": 65.6, "human": 81.6, "arch": 102.1, "results": 139, "end": 157.2}
# the team line is captioned by hand: the transcriber cannot spell the names
MANUAL = [(34.35, 38.6, "Built by Favour Olaboye and Opeyemi Mobolaji.")]
FIX = {"Gauge-less": "Gaugeless", "Gauge-less,": "Gaugeless,", "OpenCV5": "OpenCV 5", "OpenCV-5": "OpenCV 5", "CV-5": "CV 5",
       "meters": "metres", "stabilizes": "stabilises", "optimized": "optimised", "AW,": "AWS,", "Sclips": "clips",
       "Qthrough": "queue through", "Cool,": "COOL,", "graviton": "Graviton", "graviton,": "Graviton,", "step": "Step", "functions": "Functions",
       "open": "Open"}
groups = []
for k, off in OFF.items():
    ws = json.load(open(f"work/vo/{k}.tx.json"))
    words = []
    for i, w in enumerate(ws):
        t = FIX.get(w["text"], w["text"])
        if t == "Open" and i + 1 < len(ws) and ws[i + 1]["text"].startswith("CV"):   # "open CV-5" -> "OpenCV 5"
            ws[i + 1]["text"] = "OpenCV" + ws[i + 1]["text"][2:].replace("-", " "); continue
        words.append((t, w["start"] + off, w["end"] + off))
    i = 0
    while i < len(words):
        p = next((j for j in range(i, min(i + 9, len(words))) if re.search(r"[.,:;!?]$", words[j][0]) and j - i >= 1), None)
        j = p if p is not None else min(i + 5, len(words) - 1)
        if len(words) - 1 - j <= 1: j = len(words) - 1          # no one-word orphans at the end of a line
        groups.append(words[i:j + 1]); i = j + 1
groups += [[(t, a, b)] for a, b, t in MANUAL]
groups.sort(key=lambda g: g[0][1])
html = []
for i, g in enumerate(groups):
    s = g[0][1] - 0.05
    nxt = groups[i + 1][0][1] - 0.05 if i + 1 < len(groups) else g[-1][2] + 0.6
    e = min(nxt, g[-1][2] + 0.7)
    txt = " ".join(w[0] for w in g).replace(" ,", ",")
    html.append(f'    <div id="cap{i}" class="clip cap" data-start="{s:.2f}" data-duration="{e - s:.2f}"><span>{txt}</span></div>')
open("compositions/captions.html", "w").write(f"""<template>
  <style>
    @font-face {{ font-family: "Excalifont"; src: url("assets/fonts/excalifont.woff2") format("woff2"); }}
    #root {{ position: absolute; inset: 0; pointer-events: none; }}
    .cap {{ position: absolute; left: 0; right: 0; bottom: 64px; text-align: center; }}
    .cap span {{ display: inline-block; max-width: 1500px; padding: 10px 26px 12px; border-radius: 14px; background: rgba(255,255,255,0.94);
                border: 2px solid #1e1e1e; color: #1e1e1e; font-family: "Excalifont", sans-serif; font-size: 40px; line-height: 1.2; }}
  </style>
  <div id="root" data-composition-id="captions" data-width="1920" data-height="1080" data-duration="163">
{chr(10).join(html)}
  </div>
  <script>
    const tl = gsap.timeline({{ paused: true }});
    window.__timelines["captions"] = tl;
  </script>
</template>
""")
print(len(groups), "caption groups")
for g in groups[:60]: print(f"{g[0][1]:.1f}", " ".join(w[0] for w in g))
