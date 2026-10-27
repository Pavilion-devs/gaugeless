"""Demo scenes: the captured dashboard run + a chip, hand-drawn rings, notes and a punch-in. Coordinates are in
1920x1080 capture pixels; times are scene seconds (the clip starts at 0)."""
SCENES = {
 "shake": dict(dur=26.3, color="#1971c2", chip="1 &middot; Repair what it can",
   rings=[(505, 138, 740, 60, 3.9, 6.2), (738, 391, 176, 46, 9.4, 13.0), (1360, 668, 528, 68, 9.9, 14.2)],
   notes=[("sh-n1", "repair checked: camera 14.3 &rarr; 0.04 px", 1180, 748, 10.8, 14.2),
          ("sh-n2", "gauged by current meter: 1.321 m&sup3;/s &middot; error &minus;0.4%", 470, 860, 23.0, 26.3)],
   punch=(18.6, 865, 820, 1.45)),
 "refuse": dict(dur=16, color="#e03131", chip="2 &middot; Refuse what it can't",
   rings=[(14, 368, 358, 80, 1.3, 4.5), (436, 318, 290, 42, 8.4, 11.0), (1360, 596, 528, 132, 9.5, 11.0)],
   notes=[("rf-n1", "advice: seed the surface, or film when debris floats", 470, 830, 12.4, 16)],
   punch=(11.0, 865, 800, 1.45)),
 "human": dict(dur=20.3, color="#e8590c", chip="3 &middot; A human approves every alert",
   rings=[(1343, 140, 180, 56, 1.3, 5.4), (1541, 142, 172, 56, 3.1, 5.4), (1124, 504, 182, 64, 14.9, 16.6)],
   notes=[("hm-n1", "Claude plans &middot; the tools enforce the limits", 412, 120, 6.0, 13.6),
          ("hm-n2", "flood alert sent only after a person approves", 412, 372, 16.8, 20.3)],
   punch=None),
}
for k, c in SCENES.items():
    rings = "\n".join(f"      const r{i} = SK.rect(S, {x}, {y}, {w}, {h}, {{ stroke: C, strokeWidth: 5, roughness: 1.6, bowing: 1.4 }});"
                      for i, (x, y, w, h, a, b) in enumerate(c["rings"]))
    ranim = "\n".join(f"      SK.on(tl, r{i}, {a}, 0.5, \"power3.out\"); tl.to(r{i}, {{ opacity: 0, duration: 0.3 }}, {b - 0.3});"
                      for i, (x, y, w, h, a, b) in enumerate(c["rings"]))
    notes = "\n".join(f'      <div id="{i}" class="note" style="left:{x}px;top:{y}px">{t}</div>' for i, t, x, y, a, b in c["notes"])
    nanim = "\n".join(f'      tl.fromTo("#{i}", {{ opacity: 0, y: 10 }}, {{ opacity: 1, y: 0, duration: 0.35, ease: "power3.out" }}, {a});'
                      + (f' tl.to("#{i}", {{ opacity: 0, duration: 0.3 }}, {b - 0.3});' if b < c["dur"] else "")
                      for i, t, x, y, a, b in c["notes"])
    punch = ""
    if c["punch"]:
        t, cx, cy, s = c["punch"]
        # rings live in capture coordinates, so they ride along with the camera
        punch = f'      tl.to("#{k}-cam", {{ x: {round(960 - cx * s)}, y: {round(470 - cy * s)}, scale: {s}, duration: 1.1, ease: "power2.inOut" }}, {t});'
    open(f"compositions/{k}.html", "w").write(f"""<template>
  <style>
    @font-face {{ font-family: "Excalifont"; src: url("assets/fonts/excalifont.woff2") format("woff2"); }}
    #root {{ position: absolute; inset: 0; background: #0f141a; overflow: hidden; font-family: "Excalifont", sans-serif; }}
    #{k}-cam {{ position: absolute; left: 0; top: 0; width: 1920px; height: 1080px; transform-origin: 0 0; }}
    #{k}-cam video {{ position: absolute; inset: 0; width: 1920px; height: 1080px; }}
    #{k}-svg {{ position: absolute; inset: 0; width: 1920px; height: 1080px; overflow: visible; }}
    #{k}-chip {{ position: absolute; left: 0; right: 0; top: 10px; text-align: center; }}
    #{k}-chip span {{ display: inline-block; padding: 6px 24px 8px; background: #ffffff; border: 3px solid {c["color"]}; color: {c["color"]};
                    border-radius: 14px; font-size: 36px; }}
    #root .note {{ position: absolute; padding: 6px 18px 8px; background: #ffffff; border: 3px solid {c["color"]}; color: #1e1e1e;
                  border-radius: 12px; font-size: 32px; white-space: nowrap; }}
  </style>
  <div id="root" data-composition-id="{k}" data-width="1920" data-height="1080" data-duration="{c["dur"]}">
    <div id="{k}-frame" data-layout-allow-overflow>
      <div id="{k}-cam">
        <video id="{k}-video" class="clip" src="assets/clips/{k}.mp4" data-start="0" data-duration="{c["dur"]}" muted playsinline></video>
        <svg id="{k}-svg" viewBox="0 0 1920 1080" xmlns="http://www.w3.org/2000/svg"></svg>
      </div>
      <div id="{k}-chip"><span>{c["chip"]}</span></div>
{notes}
    </div>
  </div>
  <script>
    (function () {{
      const S = "#{k}-svg", C = "{c["color"]}";
{rings}
      const tl = gsap.timeline({{ paused: true }});
      tl.fromTo("#{k}-cam", {{ opacity: 0 }}, {{ opacity: 1, duration: 0.3, ease: "power2.out" }}, 0);
      tl.fromTo("#{k}-chip span", {{ opacity: 0, y: -12 }}, {{ opacity: 1, y: 0, duration: 0.4, ease: "back.out(1.8)" }}, 0.3);
{ranim}
{nanim}
{punch}
      tl.to("#{k}-frame", {{ opacity: 0, duration: 0.3, ease: "power2.in" }}, {c["dur"] - 0.3});
      window.__timelines["{k}"] = tl;
    }})();
  </script>
</template>
""")
print("ok")
