# Gaugeless — technical report

**River discharge from video, for rivers nobody gauges.** OpenCV AI Competition 2026.
Repo: <https://github.com/Pavilion-devs/gaugeless> · every number below has a command and a results file in `results/`.

---

## 1. Problem

Most of the world's rivers have no gauge. Gauging stations cost tens of thousands of dollars to build and need
regular in-river measurements to stay calibrated, so they cluster on large rivers in rich countries. The small,
fast rivers that cause flash floods are usually unmeasured.

A camera can measure a river. *Image velocimetry* tracks the water surface between frames, turns pixel motion
into metres per second, and, with a surveyed cross-section, into discharge (m³/s). The method is well
established in hydrology. What stops it being used unattended is **judgement**: the same code that is accurate
on a good clip returns a confident wrong number on a shaky, flickering, glare-covered or texture-less one, and
it never says so. In our baseline, three of four standard methods returned near-zero velocities (−95 to −100%
error) on unseeded water with no warning at all.

The dataset's own authors name this as the open problem. Perks et al. (2025, HESS 29:3727), after running
automated velocimetry on 11,000 videos, list as future work identifying seeding-density metrics to evaluate whole
videos and "eliminate poorly seeded image sequences". **That is what Gaugeless does**: an agent that diagnoses
each clip, repairs what can be repaired, measures, checks the physics, and refuses or asks a human when the
number can't be trusted.

**Users.** Hydrological agencies and NGOs that run cameras on ungauged rivers; flood-warning operators who need
a number they can act on; researchers processing large video archives.

---

## 2. Architecture

![Gaugeless architecture](docs/architecture.png)

```
clip ─► frame QA ─► diagnose ─► repair ─► measure ─► check ─► publish | refuse | escalate to a human
```

Three layers, all in this repo:

1. **Vision pipeline (OpenCV 5).** Velocimetry, repairs and orthorectification. `gaugeless/methods.py`,
   `gaugeless/ortho.py`, `gaugeless/discharge.py`.
2. **Agent.** 13 tools with guard rails (`gaugeless/agent/tools.py`) and three interchangeable planners that can
   act only through those tools: deterministic rules, Claude on Amazon Bedrock, and Qwen running inside OpenCV's
   own DNN engine.
3. **AWS.** One CloudFormation stack (`infra/gaugeless.yaml`), deployed and tested live:
   S3 upload → SQS → Graviton (arm64) worker → DynamoDB; escalations and flood alerts go to a Step Functions
   workflow that waits for a human (`waitForTaskToken`) before SNS sends anything. CloudWatch dashboard and a
   budget alarm.

The same tools are also exposed through an **MCP server** (`gaugeless/mcp_server.py`) and a **web app**
(`gaugeless/web/`) that streams each agent decision live next to the velocity overlay.

---

## 3. How OpenCV 5 is used

OpenCV does all of the vision work; nothing else touches pixels.

| Stage | OpenCV 5 functions |
|---|---|
| Tracking velocimetry (KLT-IV) | `goodFeaturesToTrack`, `calcOpticalFlowPyrLK` with forward–backward check |
| Correlation velocimetry (LSPIV) | `matchTemplate` (normalised cross-correlation), sub-pixel peak fit |
| Space-time velocimetry (STIV) | space-time image along search lines, 2-D FFT power spectrum read with `remap` along candidate slopes |
| Dense flow | `DISOpticalFlow` |
| Camera solve from ground control points | `calibrateCamera` (one view, radial k1), `projectPoints`, `Rodrigues` |
| Orthorectification onto the water plane | `remap` with precomputed maps (4 ms/frame) |
| Grid registration | `SIFT_create`, `BFMatcher.knnMatch`, `estimateAffinePartial2D` (RANSAC) |
| Stabilisation | `estimateAffinePartial2D` (RANSAC) on bank tracks, `warpAffine` |
| Flicker / glare repair | per-frame brightness equalisation + temporal-median background removal (`normalize`); glare masks with `dilate`, `morphologyEx`, `connectedComponentsWithStats` |
| Local planner | `dnn.readNetFromONNX(ENGINE_NEW)`, `enableKVCache` / `resetKVCache`, `dnn.Tokenizer` |

Runtime: `opencv-contrib-python-headless==5.0.0.93`, reported by the app itself (`cv2.__version__` is written
into every result record). The arm64 wheel already includes KleidiCV.

**Speed (laptop, Apple arm64, 464×505 frames):** KLT 14.3 ms/frame, LSPIV 3.3, STIV 0.7, DIS 12.9,
stabilisation 4.5; whole agent run on a 19.8 s clip in 2.6 s (7.5× real time). **On AWS** (t4g.small Graviton,
2 vCPU burstable): about 1.1 s per frame including the agent's repeated passes, about one minute per job.

---

## 4. The agent

### 4.1 Tools and guard rails

A planner never touches data directly. It calls tools, and the tools enforce limits a planner can't talk its way
past:

| Tool | Guard rail |
|---|---|
| `diagnose_clip` | Reports tracer density, moving fraction, displacement, flicker, glare, camera motion, flow jitter, direction coherence |
| `measure_velocity` | **Refuses** an untrackable surface (moving fraction < 0.02 or tracer density < 0.05 /m²) |
| `stabilise`, `stabilise_from_flow` | **Check themselves**: kept only if residual ≤ 1 px, fit failures ≤ 30% and camera motion actually falls; otherwise frames are restored |
| `remove_repeated_frames` | Fixes the time step (a re-encoded clip otherwise doubles every speed) |
| `compute_discharge` | Refuses without a surveyed section; fails on coverage < 60%, Froude > 1.5, or two-section continuity mismatch |
| `publish` | Refuses after a refusal; refuses until gauge points are sampled and sections computed; refuses a failed discharge or **any flood alert without human approval** |

The MCP server has no `approve` tool, so no agent can approve itself.

### 4.2 Planners

- **Rules** (`agent/rules.py`): deterministic; every threshold comes from a measurement in `plan.md`.
- **Claude Sonnet 4.5 on Bedrock** (`agent/llm.py`): Converse API with tool use, up to 20 turns, with the
  measured thresholds in its system prompt and no site-specific code.
- **Qwen2.5 in `cv2.dnn`** (`agent/local.py`): runs offline at 14–22 tokens/s on a laptop CPU.

### 4.3 What the repairs are based on

Each repair was added because the evaluation showed a failure, and kept only if it measurably helped:

- **Flow-jitter shake detector:** clean clips score ≤ 0.84 px/frame on all 11 sites (even with flicker or blur
  added); 2 px of shake scores ≥ 2.5. Threshold 1.5.
- **Bank stabilisation** found 7 px of real drift in the *unaltered* Noce drone footage. Removing it took Noce
  from 22.7% to 14.4% error, and La Vence from 16.9% to 14.9%.
- **Flow-based stabilisation** for clips with no visible bank compares each vector with the median motion in its
  own grid cell. It leaves clean clips unchanged (La Morge 0.498 → 0.497 m/s).
- **Two-section continuity:** the same water passes both sections, so their discharges must agree. On La Morge this
  check flags LSPIV (39% disagreement) while passing KLT (10%), **with no reference data at all**. That's the
  trust signal for a river nobody has gauged.

---

## 5. Evaluation

Data: Perks et al. 2020, *Towards harmonisation of image velocimetry techniques*, ESSD 12:1545, CC0. Real rivers
in 7 countries with in-river reference measurements: current meter, ADCP, radar. All 3,025 files used are listed
and size-checked in `data/perks/MANIFEST.tsv`.

### 5.1 Discharge from video

Rules agent, velocity index α = 0.85 (the textbook value, **not fitted** to any site):

| Site | Camera | Gaugeless Q (m³/s) | Reference Q (m³/s) | Error |
|---|---|---|---|---|
| La Morge (FR) | pole camera | 1.316 | 1.321, current-meter gauging | **−0.4%** |
| Thalhofen (DE) | fixed camera | 52.92 | 52.38, ADCP transect | **+1.0%** |
| Murg (CH) | drone, 4K | 2.854 | 2.76, stated with the ADCP data | **+3.4%** |
| La Vence (FR) | smartphone | 1.058 | 1.179, ADCP transect | **−10.2%** |
| Castor (CA) | bridge camera, **raw oblique video** | 25.5 | 29.6, ADCP transect | **−13.8%** |

For ADCP sites the agent uses only the ADCP's depths and positions (the cross-section survey); the ADCP's
velocities are used only for scoring. Castor is orthorectified by Gaugeless itself from raw video: a camera solve
from 12 ground control points with a radial lens term (2.6 px RMS; leave-one-out median 2.5 px, against 16 px for
a plain pinhole), projection onto the water plane at the recorded stage, one `cv2.remap` per frame. Our metric
scale differs from the dataset's published grid by 3–6%, which may explain part of the Castor gap.

### 5.2 Baseline: no single method wins

Fixed parameters, no agent, point error (MAPE) vs reference (`results/baseline_v3.jsonl`):

| Case | KLT | LSPIV | STIV | DIS |
|---|---|---|---|---|
| Arrow drone, seeded | **15.6%** | 16.9% | 20.1% | 69.5% |
| Arrow drone, unseeded | no vectors | 99.7% (silent) | 94.7% (silent) | 99.8% (silent) |
| Noce | **23.6%** | 32.9% | 99.5% | 27.0% |
| Thalhofen | **20.0%** | 33.8% | 28.3% | 36.0% |
| La Vence | **16.9%** | 18.1% | 100% | 26.4% |
| Murg | **23.1%** | 49.0% | 93.4% | 81.0% |
| Bradano | **11.2%** | 88.2% | 166.9% | 35.4% |

KLT with the KLT-IV filters is best on 9 of 11 cases, but its coverage varies (43–98% of reference points),
and every method fails silently on unseeded water. A usable-vs-unusable gap of about 1,000× in tracer density
(16.7 vs 0.007 moving features/m²) makes that failure detectable from the clip itself.

### 5.3 Agent vs fixed pipeline on damaged clips

7 rivers × {clean, shake ×3, flicker ×3, blur ×3, glare ×3, repeated frames ×2} = **105 runs**
(`results/degrade_v3.jsonl`, `degrade_v3b.jsonl`; `python -m gaugeless.agent.summarise …`). A *silent failure* is a
published result over 40% error **and** 15 points worse than the same clip undamaged.

| | Fixed pipeline | Agent |
|---|---|---|
| **Silent failures** | **24** | **0** |
| More accurate than the other, same points | — | 45 runs (56 equal within 2 points, 4 worse by ≤ 1.1 points) |
| Escalated to a human instead of publishing | never | 3 (heavy glare: discharge checks failed) |

| Damage | Fixed pipeline error | Agent error |
|---|---|---|
| Shake 5 px | 73–208% | 11–24% |
| Shake 10 px | 179–472% | 12–24% |
| Repeated frames ×2 | +99 to +123% | unchanged from clean |
| Repeated frames ×3 | +194 to +235% | unchanged from clean |

### 5.4 Planner comparison (same tools, 14 rivers)

| | Rules | Claude Sonnet 4.5 (Bedrock) | Qwen2.5 0.5B / 1.5B in `cv2.dnn` |
|---|---|---|---|
| Same publish/refuse decision as rules | — | **14 / 14** | — |
| Refuses unseeded clips | yes | yes | **no** |
| Correct repair choice (6 scenarios) | 6/6 | — | 1–3 / 6 |
| Time per clip | 2–25 s | 15–60 s | 0.7 s per decision |

Claude matches the rules planner with no site-specific code; point accuracy is within a few points either way
(better on Bradano, Thalhofen, Murg; worse on Noce, Brenta, La Vence). Once, on Tiber, Claude tried to publish
without measuring the gauge points. The tool refused, and that check is now a permanent guard rail with a test.
**Small local models can't plan this task**, so Qwen-in-OpenCV writes offline field reports instead.

### 5.5 Cloud run (live AWS, Oct 20)

| Job (uploaded to S3) | Outcome on the Graviton worker | Time |
|---|---|---|
| La Morge | published, Q 1.316 m³/s (−0.4%, identical to laptop) | 54 s |
| Noce | published | 80 s |
| Arrow unseeded | **refused**: no trackable surface | 11 s |
| La Morge, flood threshold 1.0 m³/s | **escalated** → approval email → human *rejected* → recorded, **no alert sent** | 53 s |
| La Morge, flood threshold 1.0 m³/s | **escalated** → approval email → human *approved* → SNS alert sent → recorded | 53 s |

### 5.6 Mistakes the evaluation caught (and the fixes)

1. Filling coverage gaps with LSPIV made three rivers worse (Bradano 9% → 64%). Replaced with a wide-radius KLT
   fill flagged as low-confidence.
2. The first stabiliser accepted a poor bank fit on Murg (4–12 px residual) and doubled its error. It now checks
   itself and restores the frames.
3. La Morge under 10 px of shake published an 87% error **and passed continuity**, because shake corrupts both
   sections alike. Fixed with the flow-jitter detector and flow-based stabilisation.
4. A detrend in the flow stabiliser silently changed a clean clip (Tiber 1.28 → 1.83 m/s). Removed.
5. A scoring bug in our own evaluation compared two rivers' discharge against the wrong reference. Fixed and rerun.

---

## 6. Limitations (what this does not establish)

- **Tiber (flood, radar reference):** all four methods agree on 1.2–1.3 m/s against the radar's 2.33 m/s. We
  report it as unresolved and did not tune toward it. It's consistent with tracking slow standing waves in a flood.
- **Oblique GoPro footage (Arrow)** stays at about 41% error: the far field is under-resolved after
  orthorectification.
- **Blur** is detected but not repaired (it raises error to 21–32%).
- **Discharge needs a surveyed cross-section.** Without one, Gaugeless reports surface velocity only.
- **α = 0.85 is assumed.** It's the standard value (Perks 2025 measured 0.87 ± 0.07), but sites differ.
- **Evaluation is on one dataset.** 14 clips from 7 countries, short clips (48–600 frames), no night or rain footage.
- **No COOL benchmark.** The AWS account is on the free plan, which can't subscribe to Marketplace products.
- Damage in §5.3 is simulated on real clips; real-world shake and glare can differ.

## 7. Responsible use

- **A human approves every flood alert.** The tools make it impossible for any planner, including an LLM, to send
  one without approval, and the MCP interface has no approval tool.
- **Refusal is a feature.** Gaugeless says "I can't measure this" rather than publishing a guess; the evaluation
  counts silent failures as the main metric, not average accuracy.
- **Not an operational warning system.** It has been validated against reference measurements, not run in
  production. A deployment would need site calibration and local hydrologists in the loop.
- **Privacy:** cameras pointed at rivers can capture people. Clips are stored encrypted in a private bucket and
  expire after 30 days; only velocity fields and numbers are kept.
- **Data:** Perks et al. 2020 is CC0; models are Qwen2.5 (Apache-2.0) and Claude via Amazon Bedrock.

---

## 8. Reproduce

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
make test                       # 25 tests, synthetic, no dataset needed
python -m gaugeless.discharge   # La Morge: video -> m³/s at both sections
make agent degrade              # agent on every river; agent vs fixed on 105 damaged clips
python -m gaugeless.agent.run lamorge --planner claude      # needs AWS creds + Bedrock
make deploy EMAIL=you@example.com && make upload-demo-data upload-code   # AWS
```

## References

- Perks, M. T. et al. (2020). Towards harmonisation of image velocimetry techniques for river surface velocity
  observations. *Earth System Science Data* 12, 1545–1559. Dataset DOI 10.4121/uuid:014d56f7-06dd-49ad-a48c-2282ab10428e.
- Perks, M. T. et al. (2025). Unsupervised image velocimetry. *Hydrology and Earth System Sciences* 29, 3727.
- Tauro, F. et al. (2017). Measurements and observations in the XXI century (MOXXI). *Hydrological Sciences Journal*.
