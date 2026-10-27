# Review: Gaugeless demo (2:43, 1920×1080, 30 fps)

## Structure
| Time | Scene | Source |
|---|---|---|
| 0:00 | Hook: rivers with no gauge | illustrated |
| 0:12 | The catch: confident wrong numbers | illustrated |
| 0:27 | Title | illustrated |
| 0:34 | The team: Favour Olaboye (AI Engineer), Opeyemi Mobolaji (ML Engineer & Developer) | illustrated |
| 0:39 | 1 · Repair: La Morge, 10 px shake → 1.316 m³/s (gauged 1.321) | live dashboard capture, http://100.63.108.38/ |
| 1:05 | 2 · Refuse: Arrow unseeded → refused | live dashboard capture |
| 1:21 | 3 · Human approval: Claude on Bedrock, flood threshold → approve | live dashboard capture |
| 1:42 | How it works: OpenCV 5 pipeline → planner → AWS → COOL | illustrated build |
| 2:19 | Results: 5 rivers, 24 vs 0 | illustrated |
| 2:37 | End card | illustrated |

## Checks
- `hyperframes check`: passes. One warning kept on purpose: the architecture's small detail text drops to
  about 20 px in the zoomed-out view (box titles stay readable). Same trade-off as the PRIME reel.
- Every number on screen comes from the README / results files or the captured runs themselves.
- Demo captures are real runs on the live AWS endpoint. The waiting time is sped up (up to 9.5×), and the
  cuts land on the narration. Clicks and results are shown at about real speed.

## Audio
- Voice: Kokoro `am_michael` (local), normalised to −16 LUFS. The team line spells the names phonetically for the voice; its caption is set by hand in `build_captions.py`. HeyGen's free voice allowance ran out partway,
  so every line uses the same local voice.
- Music: HeyGen catalog underscore (not committed: catalog track), looped with 3 s crossfades, about −32 LUFS, gain 0.55. Re-fetch with `npx hyperframes media-use resolve --type bgm` and loop it into `assets/bgm.wav`.

## Rebuild
- `python3 work/build_captions.py`: captions from the VO word timings.
- `python3 work/build_demos.py`: the three demo scenes (rings, notes, punch-ins).
- `node work/capture.js shake work/cap/shake.mp4` re-captures a run (needs `npm i puppeteer-core`; CDP screencast at 1920×1080).
