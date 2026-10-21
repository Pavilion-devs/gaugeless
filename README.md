# Gaugeless

**River discharge from video, for rivers nobody gauges.** An agent built on OpenCV 5 diagnoses a clip,
repairs what it can (camera shake, re-encoded frame rates, flicker, glare), and measures surface velocity.
It turns that into discharge in m³/s at surveyed cross-sections, checks the physics, and **refuses or asks a
human** when the number can't be trusted.

Most of the world's rivers have no gauge. A phone, a drone or a fixed camera can measure them, but only if
someone can tell a good measurement from a confident wrong one. That judgement is what Gaugeless automates.

## Results so far (real rivers, in-river reference measurements)

Data: Perks et al. (2020), *Towards harmonisation of image velocimetry techniques for river surface velocity
observations*, ESSD 12:1545 (CC0). It has 14 sites in 7 countries with current-meter, ADCP and radar references.

**Discharge from video, rules agent, α = 0.85 (not fitted to any site):**

| Site | Camera | Agent Q (m³/s) | Reference Q (m³/s) | Error |
|---|---|---|---|---|
| La Morge (FR) | pole camera | 1.316 | 1.321, current-meter gauging | **−0.4%** |
| Thalhofen (DE) | fixed camera | 52.92 | 52.38, ADCP transect | **+1.0%** |
| Murg (CH) | drone, 4K | 2.854 | 2.76, stated with the ADCP data | **+3.4%** |
| La Vence (FR) | smartphone | 1.058 | 1.179, ADCP transect | **−10.2%** |
| Castor (CA) | bridge camera, **raw oblique video** | 25.5 | 29.6, ADCP transect | **−13.8%** |

Castor is orthorectified by Gaugeless itself from the raw video:
1. A camera solve from 12 ground control points with a radial lens term (2.6 px RMS; the point left out of each fit
   lands within 2.5 px median, against 16 px for a plain pinhole).
2. Projection onto the water plane at the recorded stage.
3. One `cv2.remap` per frame (4 ms).

Our metric scale differs from the dataset's published grid by 3–6%, which may explain part of the Castor gap.

For Thalhofen and La Vence, the reference Q is integrated from the ADCP verticals (depth-averaged velocity ×
depth); the agent uses only the ADCP depths and positions, never its velocities.

**Agent vs a fixed pipeline on damaged real clips** (`python -m gaugeless.agent.summarise results/degrade_v3.jsonl
results/degrade_v3b.jsonl`):
- 105 runs: 7 rivers × {clean, shake ×3, flicker ×3, blur ×3, glare ×3, repeated frames ×2}.
- A *silent failure* is a published result over 40% error and 15 points worse than the same clip undamaged.

| | Fixed pipeline | Agent |
|---|---|---|
| Silent failures | **24** | **0** |
| More accurate than the other, on the same points | — | 45 runs (56 equal within 2 points, 4 worse) |
| Escalated to a human instead of publishing | never | 3 (heavy glare: discharge checks failed) |

| What | Result |
|---|---|
| La Morge's two cross-sections | each within 5% of the gauging, and within 10% of each other |
| The two-section continuity check (needs no reference) | flags LSPIV on La Morge (39% disagreement) while passing KLT |
| Unseeded clips (no tracers) | **refused**; fixed pipelines return near-zero speeds (−95 to −100%) with no warning |
| Camera shake (5 px) | fixed pipeline 115–208% error → agent 11–24% after self-checked stabilisation |
| Re-encoded frame rate (each frame ×2) | fixed pipeline +103 to +123% → agent unchanged from clean |
| Noce (drone; no damage added) | 22.7% → 14.4%: the agent finds 7 px of camera drift and removes it |

Full tables: [`plan.md`](plan.md) and `results/*.jsonl`. Every number has a command (see below).

## How it works

```
clip ─► frame QA ─► diagnose ─► repair ─► measure ─► check ─► publish | refuse | escalate to a human
         size,        tracer      stabilise   KLT (PyrLK +   continuity
         repeated     density,    (bank       KLT-IV         between two
         frames       motion,     RANSAC,     filters),      sections,
                      flicker,    self-       LSPIV, STIV,   Froude,
                      glare,      checking),  DIS            coverage
                      camera      dedupe,                    floor,
                      motion      glare mask,                per-point
                                  background                 confidence
```

- **Tools with guard rails** (`gaugeless/agent/tools.py`). Planners can only act through tools, and the tools
  enforce the limits: they won't measure an untrackable surface, won't publish a failed discharge, and won't
  send a flood alert without human approval.
- **Three planners, same tools:**
  - `rules`: deterministic; every branch is tied to a measurement in `plan.md`.
  - `claude`: Claude on Amazon Bedrock with tool use.
  - `local`: Qwen2.5-0.5B running inside **OpenCV 5's own DNN engine** (`cv2.dnn` + `cv2.dnn.Tokenizer`), for
    offline/edge use.
- **Planner comparison, 14 rivers:** Claude made the same publish/refuse decision as the rules planner on 14 of
  14, with point accuracy within a few points either way, and with no site-specific code. When Claude once tried to
  publish without measuring the gauge points, the tool refused, and that check is now a guard rail.
- **MCP server** (`python -m gaugeless.mcp_server`) exposes the same tools to any MCP client. It has no
  `approve` tool, so an agent can't approve itself.
- **AWS** (`infra/gaugeless.yaml`): S3 upload → SQS → Graviton worker (stock OpenCV 5 or COOL) → DynamoDB.
  Escalations and flood alerts go to Step Functions, which waits for a human (`waitForTaskToken`, approve/reject
  links by email) before SNS sends anything. CloudWatch shows ms/frame and agent outcomes.
- **Live on AWS (Oct 20):** jobs uploaded to S3 run on the Graviton worker (aarch64, OpenCV 5.0.0); La Morge
  reproduces −0.4% in the cloud, the unseeded clip is refused, and a flood-threshold job waited for a human: one
  test was rejected (no alert sent), one approved (alert sent), both recorded in DynamoDB.
- **Web app** (`python -m gaugeless.web.server`): pick a river, optionally damage the clip, and watch each agent
  decision stream in next to the velocity overlay; approve escalations as the operator.

## Run it

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
make test                       # 25 tests, synthetic, no dataset needed
python -m gaugeless.discharge   # La Morge: video -> m³/s at both sections
python -m gaugeless.agent.run lamorge noce arrow_uas_unseeded            # rules planner, with trace
python -m gaugeless.agent.run lamorge --planner claude                    # needs AWS creds + Bedrock
python -m gaugeless.agent.degrade_eval && python -m gaugeless.agent.summarise results/degrade.jsonl
python -m gaugeless.web.server --port 8080
make deploy EMAIL=you@example.com && make upload-code upload-demo-data   # AWS
```

Dataset: the 33 GB archive is at <https://doi.org/10.4121/uuid:014d56f7-06dd-49ad-a48c-2282ab10428e>. Put the
site folders under `data/perks/` and run `python -c "from gaugeless.cases import verify_downloads as v; print(v())"`
to check every file against `data/perks/MANIFEST.tsv`.

## Honest limits

- **Tiber (flood, radar reference):** all four methods agree on 1.2–1.3 m/s against a radar 2.33 m/s. We report
  this as unresolved and haven't tuned toward it.
- Oblique GoPro footage (Arrow) stays at about 41% error: the far field is under-resolved after orthorectification.
- Blur is detected but not repaired; it raises error moderately (to 21–32%).
- **Small local models can't plan this.** Qwen2.5 0.5B and 1.5B run inside OpenCV 5's DNN engine (KV cache,
  `cv2.dnn.Tokenizer`, 14–22 tokens/s generation on a laptop CPU), but they choose the right repair in only 1–3 of 6
  scenarios and miss "refuse an unseeded clip". The local model writes offline field reports; Claude on
  Bedrock plans.
