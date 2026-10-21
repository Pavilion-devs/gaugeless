# GAUGELESS — river discharge from video, for rivers nobody gauges

> Working name. OpenCV AI Competition 2026 (powered by AWS).
> **Deadline: Mon Oct 26 2026, 11:45 PM PDT** (Devpost header; body says 11:59, plan for the earlier one).
> Today: Mon Oct 19. Seven build days.

---

## 1. The goal

Win 1st overall and both special awards (Best Use of COOL, Agentic Vision). The entry to beat is
**TremorCV** (TusharTechs/tremorcv): vibration from handheld video, COOL benchmarks on Graviton4,
MCP server, live endpoint, honest report. Matching its rigor is the floor, not the goal.

### Where TremorCV is weak (from its own report) and how we beat it

| TremorCV gap | Gaugeless answer |
|---|---|
| Real ground truth = 1 phone clip of a **laptop test pattern** + 1 fan with no tachometer | **12 real rivers, 7 countries**, scored against ADCP / current-meter / radar measurements made in the water |
| Agent eval is on **simulated** machines (n=80) | Agent eval on **real footage**, plus controlled degradations of real clips where the clean clip's result is known |
| S3/Lambda/SQS **drawn but not wired**; one EC2 box | Fully wired: S3 → SQS → Graviton+COOL workers → DynamoDB → Step Functions human approval → SNS |
| Agent is a fixed ladder, no model, narrow actions | Same safety (guards live in tools) **plus** a planner that chooses method, region, stride, and when to stop; three planners compared head to head |
| Output in pixels | Output in **m/s and m³/s**, the units a hydrologist acts on |
| No external baseline | Compared against **pyORC** (open-source standard) and the published KLT-IV approach on identical inputs |

### The rubric, and where each point comes from

| Criterion | Weight | Our evidence |
|---|---|---|
| Technical execution | 30 | Multi-method OpenCV 5 velocimetry, orthorectification from raw video, physics checks, tests, falsifiable numbers |
| Innovation | 20 | Method selection driven by measured surface conditions (the field's stated open problem); mass-continuity self-check; OpenCV-native LLM planner |
| Real-world impact | 20 | Most of the world's rivers are ungauged; flood warning + water resources; a $150 camera replaces a gauging station |
| User experience | 10 | Live endpoint: pick a river, watch the agent decide, see vectors / profile vs ADCP / discharge |
| Documentation | 10 | Report, architecture diagram, judge one-pager, reproduce-every-number script |
| Cloud, reproducibility, responsible ops | 10 | Real AWS pipeline, IaC, human approval before any alert, cost per video-hour |

Agentic Vision rubric: OpenCV+agent integration 30 · orchestration & autonomy 25 · task effectiveness 20 ·
failure handling / observability / security / human control 15 · UX & docs 10.

---

## 2. Research findings that shape the build

1. **The dataset's own authors name our feature as future work.** Perks et al. 2025 (HESS 29:3727,
   "Unsupervised image velocimetry", KLT-IV on 11,000 videos) say they did *not* address
   noisy lighting, low surface texture, wind and standing waves, and that future work should
   "identify and apply appropriate seeding density metrics to evaluate the quality of entire
   videos, eliminate poorly seeded image sequences ... or focus analysis on specific
   cross-sections", plus use wind information. **That is the agent.** Quote it in the report and video.
2. **No single method wins everywhere.** The literature: LSPIV systematically *underestimates*
   (Tauro 2017; AGU 2024: 26–64% normalised error with public tools); PTV wins when discrete
   tracers exist; STIV often matches best on texture-poor water. Choosing the method from what
   the camera sees is a real decision, not a decoration.
3. **KLT-IV is Lucas–Kanade tracking**, i.e. `cv2.goodFeaturesToTrack` + `cv2.calcOpticalFlowPyrLK`
   with forward–backward error checks. The state-of-the-art automated workflow maps directly onto
   OpenCV. Its filters: drop vectors >45° off the flow line and <0.1 m/s.
4. **Velocity index α (surface → depth-averaged)** is 0.85 by convention; Perks 2025 measured
   0.87 ± 0.07. ADCP profiles in our dataset (Thalhofen, Murg, La Vence) let us *measure* α per
   site instead of assuming it.
5. **OpenCV 5.0.0 (pip `opencv-contrib-python-headless==5.0.0.93`) verified locally** with
   everything we need: PyrLK, DIS, Farneback, RLOF, DualTVL1, matchTemplate, dft, phaseCorrelate,
   findHomography, solvePnP, CLAHE, `cv2.dnn.Tokenizer`. The arm64 pip wheel already includes
   KleidiCV 26.03, so our COOL comparison must be honest about what COOL adds on top.
6. **OpenCV 5.0.0 bug to avoid:** `cv2.phaseCorrelate` mutates its inputs in place (found by
   TremorCV). Copy the arrays before calling it, or use our own `cv2.dft` cross-correlation.
7. **OpenCV-native LLM:** official samples `samples/dnn/qwen_inference.py` (Qwen2.5-0.5B-Instruct,
   KV-cache) and `vlm_inference.py` (PaliGemma2-3B). No competitor uses this.
8. **COOL** ships on AWS Marketplace (Graviton2/4/5, AMI and Docker; OpenCV 5.x + KleidiCV + ARMPL;
   Python 3.10–3.12 venvs under `/opt/cool/venvs/`). 7-day free trial, then about $0.06/h usage fee.
   Its gains come from the warp, resize, filter and DFT kernels, which are the heart of our
   orthorectification and correlation stages.
9. **Competition map** (about 50 GitHub entries): crowded in factory inspection, gauges, construction,
   roads, medical, documents. One weak flood entry (FloodLens: YOLO flood segmentation, no flow
   measurement). **No river-velocity or discharge entry exists.**

---

## 3. Data

Source: Perks et al. 2020, ESSD 12:1545, "Towards harmonisation of image velocimetry techniques".
4TU.ResearchData, **CC0**, one 33 GB `data.zip`. The full download runs at about 250 KB/s
(18–36 h), so we pull individual files straight out of the remote zip with HTTP range requests
(`tools/httpzip.py`). Local copy: `data/perks/`.

| Case | Country | Platform | Reference | Status |
|---|---|---|---|---|
| Thalhofen | DE | fixed cam, 30 Hz ortho | ADCP profiles (depth × width) | pulling |
| La Vence | FR | Galaxy S7 phone, 30 Hz ortho | ADCP profiles | pulling |
| Noce | IT | drone, 12 Hz | current meter, 13 verticals | pulling |
| Tiber | IT | IP "GaugeCam" under bridge | surface radar, 1 value (2.33 m/s) | pulling |
| La Morge | FR | pole cam, 5 Hz ortho | discharge only + cross-section survey | have |
| Arrow | UK | drone + GoPro, seeded and unseeded | EM current meter | have / pulling |
| Bradano | IT | drone, 24 Hz, stabilised | current meter | pulling |
| Brenta FLIR | IT | **thermal**, ice-cube tracers | current meter, 4 pts | pulling |
| Brenta GoPro | IT | bridge GoPro, 25 Hz | current meter | 1 of 12 clips queued |
| Alpine | AT | drone, 12.5 Hz, cornstarch seeding | propeller meter, 23 pts + direction | first 360 frames queued |
| Murg | CH | drone, 12 Hz ortho + GCPs | ADCP profiles | first 360 frames queued |
| Castor (×4 videos) | CA | oblique IP cams | ADCP + ADV | **raw mp4 + GCPs**: we orthorectify ourselves |
| Salmon | CA | drone at 102 m, **glare** | ADV, 26 pts | **raw MOV + GCPs**: we orthorectify ourselves |
| St Julien | FR | storm-chaser **flash flood** | none | have (demo clip only) |

Each reference spreadsheet has its own layout, so we need one loader per case → `gaugeless/data/refs.py`,
normalised to `(x_px, y_px, v_ref, kind={surface|depth_profile|integrated|discharge})`.

**Own field validation (strongly recommended):** film a local stream or drain with a phone, lay 4 markers
with a measuring tape, and time an orange over a measured distance (USGS float method), 5 runs.
One more real site, captured by us, covering the full raw-video → ortho → discharge path.

---

### Day-0 results (Oct 19, `spikes/klt_arrow.py`, River Arrow drone footage vs 49 current-meter points)

| Clip | Pipeline | MAPE | Bias | RMSE |
|---|---|---|---|---|
| Seeded | naive KLT | 22.7% | −17.2% | 0.152 m/s |
| Seeded | KLT + KLT-IV filters (≥0.1 m/s, ±45° of flow) | **15.3%** | **+1.4%** | **0.080 m/s** |
| Unseeded | naive KLT | 89.2% | −89.2% | 0.390 m/s |
| Unseeded | KLT + filters | *no vectors survive* | | |

Reading: the fixed pipeline is decent when tracers exist and fails badly when they don't. The naive
version fails silently with a confident wrong number. That gap is the agent's job: detect low
seeding, switch method (LSPIV/STIV on texture) or request seeding, and refuse rather than report
−89%. This was on non-stabilised frames; stabilised frames are still downloading.

### Day-1 baseline: fixed parameters, no agent (`results/baseline_v1.jsonl`)

MAPE vs in-river reference (coverage = share of reference points with a measurement). 120 frames.

| Case | KLT | LSPIV | STIV | DIS |
|---|---|---|---|---|
| Arrow drone, seeded | **15.6%** (98%) | 16.9% (76%) | 20.1% (76%) | 69.5% |
| Arrow drone, unseeded | no vectors | 99.7% (silent) | 94.7% (silent) | 99.8% (silent) |
| Arrow GoPro, seeded | 40.9% | 44.5% | 46.8% | 52.4% |
| Arrow GoPro, unseeded | 37.6% (14%) | 77.5% | **57.2%, bias −2%** | 97.9% |
| Bradano | **11.2%** (43%) | 88.2% | 166.9% | 35.4% |
| Alpine | **12.4%** (57%) | 23.4% (100%) | 40.5% | 50.4% |

What this says:
1. No fixed method wins everywhere. KLT is most accurate but often covers only half the river; LSPIV covers
   everything but swings from 17% to 88%; STIV is the only one that holds bias on texture-poor GoPro water.
2. **Silent failures are the main danger.** On unseeded water, LSPIV, STIV and DIS all return confident
   near-zero velocities (−95 to −100%) with no error raised. The diagnosis stage must catch this before
   any number leaves the system.
3. Every method recovers synthetic motion within 2% (`tests/test_methods.py`), so these are real-world
   failure modes, not code bugs.
4. DIS (temporal median) underestimates everywhere; it stays as a coverage fallback only.

### Day-1 findings that feed the agent

- **Silent failures are detectable from the clip itself** (`gaugeless/diagnose.py`, first 30 frames):

  | Clip | tracer density (moving features/m²/frame) | moving fraction | direction |
  |---|---|---|---|
  | Arrow drone seeded | 16.7 | 0.31 | 18° (correct) |
  | Arrow drone unseeded | **0.007** | **0.0001** | 113° (wrong) |
  | Arrow GoPro seeded | 14.5 | 0.36 | 22° |
  | Arrow GoPro unseeded | **0.021** | **0.0004** | 128° (wrong) |

  A factor of about 1,000 separates usable clips from unusable ones. That's the first refusal rule.
- **The water region matters.** On Bradano, textured jittery banks crowd out the dark water's sparse tracers.
  Automatic masking (dynamic/texture ratio) works on 4 of 5 sites and fails on Bradano. Decision: a water
  polygon per site drawn once (standard practice; KLT-IV also requires an ROI and flow direction), plus feature
  detection inside the mask. Auto-masking stays as an agent proposal, not the default.
- **Coverage vs clip length:** KLT covers 43–57% of points on sparse-tracer rivers (Bradano, Alpine) with 120
  frames. Extending the clip is an agent action with a measurable effect.
- **Oblique far-field:** Arrow GoPro reference points in the far field lie where the orthorectified image has
  almost no resolved tracer motion. The agent should flag low-resolution regions instead of reporting them.
- Data hygiene: partially downloaded frames exist after network drops. **Done:** `data/perks/MANIFEST.tsv`
  (3,025 files) + `cases.verify_downloads()`; all 3,025 files verified complete on Oct 19.

### Day-1 complete: every case with a reference (`results/baseline_v3.jsonl`, 150 frames)

MAPE. ADCP cases are scored against α·(depth-averaged) with α = 0.85.

| Case | Reference | KLT | LSPIV | STIV | DIS |
|---|---|---|---|---|---|
| Noce (drone) | 13 surface | **23.6%** (77%) | 32.9% | 99.5% | 27.0% |
| Thalhofen (fixed) | 280 ADCP | **20.0%** (68%) | 33.8% | 28.3% | 36.0% |
| La Vence (phone) | 8 ADCP | **16.9%** | 18.1% | 100% | 26.4% |
| Murg (4K drone) | 85 ADCP | **23.1%** (74%) | 49.0% | 93.4% | 81.0% |
| Brenta FLIR (thermal) | 4 surface | 42.9% | 45.6% | 52.8% | 90.9% |
| Tiber (IP cam, flood) | 1 radar | −46.5% | −51.0% | −55.8% | −44.6% |
| Arrow GoPro (new polygon) | 49 surface | 40.9% | 44.5% | 46.8% | 52.4% |

- KLT with the KLT-IV filters is best on 9 of 11 cases; STIV only helps on texture-poor oblique water.
- The new GoPro water polygon changed nothing, which confirms the GoPro error is far-field resolution, not the mask.
- **Tiber is unresolved.** All four methods agree on 1.2–1.3 m/s against a radar 2.33 m/s. The readme fps (6.95)
  and gsd match the video, there are no duplicate frames, and a deeper pyramid or larger LSPIV search changes
  nothing. Speed rises with frame gap (1.19 → 1.61 at gap 3), then drops as tracks are lost. This is consistent
  with tracking slow quasi-standing waves at a flood. Report it as a method–reference disagreement, and don't tune
  toward it. The agent needs a rule for it: "methods agree, plausibility check (Froude) required".

### Day-2 first result: video → discharge (m³/s), `python -m gaugeless.discharge`

La Morge: two surveyed cross-sections in view, current-meter gauging Q = 1.321 m³/s.
Mid-section method, α = 0.85, gaps filled with u ∝ h^(2/3).

| Method | Section A | Section B | Continuity A vs B |
|---|---|---|---|
| KLT | 1.253 ± 0.115 (**−5.1%**) | 1.385 ± 0.115 (**+4.8%**) | 10%, pass |
| LSPIV | 0.733 (−44.6%) | 1.093 (−17.3%) | **39%, FAIL** |

**The continuity check catches the bad method with no reference data at all.** This is the headline
mechanism for the agent: compute Q at two sections, and if they disagree, switch method or region, or escalate to a human.

### Day-2 (done Oct 19 night): discharge on 4 sites, the agent, and everything around it

**Discharge from video, rules agent, α = 0.85 (not fitted):**

| Site | Agent Q | Reference Q | Error |
|---|---|---|---|
| La Morge | 1.319 | 1.321 (current meter) | −0.2% |
| Thalhofen | 52.92 | 52.38 (ADCP integrated) | +1.0% |
| Murg | 2.854 | 2.76 (stated) | +3.4% |
| La Vence | 1.058 | 1.179 (ADCP integrated) | −10.2% |

`refs.adcp_section()` turns each ADCP transect into a cross-section (pixel position, station, river depth).
The agent uses only depths and positions; the ADCP velocities are used only for scoring.

**Built:**
- `gaugeless/agent/tools.py`: 13 tools with guard rails.
- Planners:
  - `rules.py`
  - `llm.py`: Claude Sonnet 4.5 on Bedrock, Converse tool use. Works; same La Morge answer, and it correctly refuses unseeded clips.
  - `local.py`: Qwen in cv2.dnn.
- `mcp_server.py`: MCP 2.x, 15 tools, no approve tool; tested with a stdio client.
- `worker.py`: SQS → agent → DynamoDB → Step Functions.
- `infra/gaugeless.yaml`: validated by CloudFormation, **not deployed**.
- `web/`: stdlib server with SSE, live decision trace, overlays, operator approval.
- `bench/bench.py`: per-stage timings with provenance, for the COOL comparison.
- `degrade.py`, `agent/degrade_eval.py` and `agent/summarise.py`.
- `Makefile`, `README.md`, and 24 tests.

**Repairs and the measurements behind them:**
- **Bank stabilisation** (similarity transform, RANSAC). It checks itself: residual ≤ 1 px, fit failures at most 30% of frames (those
  frames get the camera motion interpolated from neighbours; they're never dropped, because dropping doubles the time step), and camera motion must fall.
  - On Murg the bank isn't rigid (fits fail on 119/150 frames), so the result is rejected and frames restored.
  - **Unaltered Noce footage has 7 px of drift.** Removing it takes Noce from 22.7% to 14.4%; La Vence goes from 16.9% to 14.9%.
- **Flow jitter**: a shake detector that needs no bank. Clean clips score ≤ 0.84 px/frame on all 11 sites (with flicker/blur added
  too); 2 px shake scores ≥ 2.5. The threshold is 1.5.
- **Flow-based stabilisation** for footage with no bank. Each vector is compared with the time-median motion *in its own grid
  cell*, over two passes. It leaves clean clips unchanged (La Morge 0.498 → 0.497) and brings shaken ones back to within
  1–13% of clean.
- **Repeated frames:** detected (consecutive frame difference < 0.25 grey levels) and the time step corrected.
  Fixed pipeline: +103% for ×2 repeats, +194–218% for ×3. Agent: equal to clean.
- **Flicker → background subtraction:** neutral or better on 8/9 rivers; Alpine −2 points of accuracy but +10 points of coverage.

**Mistakes found by the evaluation and fixed (worth telling in the report):**
1. Filling coverage gaps with LSPIV, even after an agreement check, made Bradano, Alpine and Thalhofen worse (Bradano 9% → 64%).
   Replaced it with a wider-radius KLT fill that's flagged low-confidence.
2. The first stabiliser accepted a poor bank fit on Murg (4–12 px residual) and doubled its error. It now checks itself.
3. La Morge under 10 px shake: published **87% error, and continuity passed**, because shake corrupts both sections alike.
   There was no water polygon, so no bank and no camera-motion measurement. Fix: a La Morge polygon plus the flow-jitter detector.
4. The flow stabiliser's linear detrend is the same as subtracting a constant velocity: it changed clean Tiber from 1.28 to 1.83 m/s.
   Removed it, and switched to location-matched residuals.

**Planner finding (negative, report it):**
- Qwen2.5-0.5B runs inside OpenCV 5 (`readNetFromONNX(ENGINE_NEW)` + `enableKVCache` / `resetKVCache` +
  `cv2.dnn.Tokenizer`), at 0.7 s per decision and 14–22 tokens/s generation on an M-series CPU.
- But as a decision-maker it ignores the measurements: it picks one option for every scenario, both with raw numbers and with
  PROBLEM/ok labels. **0.5B is too small to plan.** It's still useful for the offline operator report (`local.report`).
- 1.5B is downloading for the same test.

---

## 4. What we build

### 4.1 Vision pipeline (all OpenCV 5)

```
raw video ─► frame QA ─► stabilise ─► orthorectify ─► surface diagnosis ─► velocimetry ─► filter ─► velocity field
                │            │             │                 │                  │
                │   ORB/AKAZE+homography   │       texture, seeding density,   KLT/PTV, LSPIV (dft xcorr),
                │   on bank features       │       glare map, wind ripple,     STIV, DIS dense
   blur, exposure,           GCPs → solvePnP / findHomography,                 (method chosen by agent)
   dropped frames            warpPerspective at chosen m/px
                                                                         ─► velocity index α ─► cross-section ─► Q (m³/s)
                                                                                  │
                                                       physics checks: Froude plausibility, flow direction,
                                                       mass continuity between two sections, bootstrap uncertainty
```

- **Surface diagnosis** produces measured quantities, not labels: features per m² (seeding density),
  temporal-std glare mask, local texture energy, dominant ripple orientation vs flow direction (wind),
  displacement per frame at the current stride (target 2–8 px).
- **Velocimetry methods**, all on one interface `measure(frames, roi, params) -> VectorField`:
  - `klt`: Shi–Tomasi + PyrLK, forward–backward check (the KLT-IV approach)
  - `lspiv`: interrogation-window cross-correlation via `cv2.dft`, sub-pixel peak
  - `stiv`: space–time images along search lines, streak angle from the 2-D FFT / structure tensor
  - `dis`: dense DIS optical flow, a fallback on continuous texture
- **Physics checks:**
  - Froude number in a plausible range for the measured depth.
  - Vectors aligned with the channel axis.
  - **Q at section A ≈ Q at section B** (mass conservation). This is a self-check that needs no
    ground truth and works on any river; it's our headline trust signal.

### 4.2 Agent layer

The tools are exposed through an **MCP server** (`gaugeless/mcp_server.py`), and the same tools are
callable in-process. **Guard rails live in the tools:** `compute_discharge` refuses when continuity fails
beyond tolerance, when coverage is below threshold, or when no stage/cross-section is given.

| Measurement says | Agent does |
|---|---|
| displacement/frame < 1 px or > 10 px | change the frame stride, re-measure |
| low seeding density, continuous texture | switch KLT → LSPIV or STIV |
| discrete tracers present | prefer KLT/PTV |
| glare mask > X% of the ROI | mask the glare, move the ROI to an unaffected band |
| ripple orientation off the flow axis + upstream vectors | flag wind, restrict to the centre band, widen the uncertainty |
| continuity mismatch between sections > tolerance | try another method or ROI; if it persists, **refuse** |
| no stable bank features | request a new clip / tripod |
| unseeded, nothing trackable | request seeding (Arrow seeded vs unseeded proves the value) |
| confidence < threshold or flood threshold crossed | **escalate**: Step Functions waits for a human approval token before SNS sends the alert |

**Three planners, same tools, compared on the same cases** (this comparison is itself a result):
1. `rules`: a deterministic policy (the ablation floor, like TremorCV's ladder).
2. `claude`: Claude on **Amazon Bedrock**, with tool calling over the MCP surface.
3. `local`: **Qwen2.5-0.5B running inside OpenCV 5's DNN engine** (`cv2.dnn` + `cv2.dnn.Tokenizer`),
   constrained to a JSON action schema. The offline/edge story: no PyTorch, no ONNX Runtime, no internet.

Every decision is written to a trace (JSON), streamed to the UI, and stored in DynamoDB.

### 4.3 AWS (the real pipeline, not a diagram)

```
browser / camera ──► S3 (uploads) ──► SQS (jobs) ──► Graviton workers (ECS on EC2 c8g, COOL AMI/Docker)
                                                          │  OpenCV 5 pipeline + agent
                                                          ▼
                                    DynamoDB (results, traces) ◄──► API (Lambda / FastAPI on the worker)
                                                          │
                                         flood threshold ─► Step Functions (waitForTaskToken = human approval)
                                                                         └► SNS (SMS/email alert)
                                    CloudWatch metrics: ms/frame, $/video-hour, agent decisions
```
IaC with CDK or Terraform, committed. Bedrock for the `claude` planner.

### 4.4 COOL benchmark (Best Use of COOL)

- Same Graviton4 instance, same clips: stock `opencv-python 5.0.0` vs COOL.
- Per stage: ortho warp, LSPIV dft, KLT, STIV FFT. End-to-end ms/frame, ×realtime, **$ per river-hour**.
- Plus the OpenCV-native LLM planner's tokens/s on Graviton, stock vs COOL. Nobody else measures this.
- Keep raw results with provenance (instance id, AMI, cv2 build info hash), as TremorCV does.
  Report the ops that show no gain.

### 4.5 Web endpoint (UX)

- Single page: choose a river case or upload a clip (+4 GCP clicks) → watch the agent's decisions stream
  live (SSE) beside the video.
- Overlays: vectors, glare mask, sections. Charts: surface-velocity profile vs ADCP, Q with an
  uncertainty band.
- An "alert" demo that shows the human-approval step.
- HTTPS via CloudFront (TremorCV is plain HTTP).

---

## 5. Evaluation (what we will report)

1. **Point accuracy vs reference** for every case: MAPE, bias, RMSE (m/s), n points. Depth-profile
   references converted to surface velocity by profile extrapolation; α reported per site.
2. **Baselines on identical inputs:** pyORC defaults; each single method with fixed parameters;
   the agent. Table: case × method.
3. **Discharge:** La Morge (discharge reference), Castor (30.5 m³/s reference), Murg (2.76 m³/s).
4. **Agent effectiveness:** controlled degradations of real clips (blur, injected glare, frame drops,
   wrong stride, tracer removal by masking), with the clean-clip result as truth.
   - Report accuracy vs severity for agent vs fixed pipeline (a curve, not a ratio).
   - Report escalation rate, and the "refused when it should have" rate.
5. **Planner comparison:** rules vs Claude vs local Qwen, on decision agreement, final error,
   latency and cost.
6. **Continuity self-check validity:** does continuity mismatch predict error vs reference? If yes,
   it's a trust signal that works on ungauged rivers. That is the key claim.
7. **Failures, documented:** St Julien (no reference), heavy glare, unseeded Arrow.

Every number gets a script: `make eval` reproduces the report tables.

---

## 6. Schedule

| Day | Date | Deliverable |
|---|---|---|
| 0 | Mon 19 | Research, data pull, plan (this file). **First number:** KLT on 2–3 small cases vs reference |
| 1 | Tue 20 | Reference loaders for all cases; KLT + LSPIV + STIV + DIS on one interface; first case × method table — **done Oct 19** |
| 2 | Wed 21 | Surface diagnosis, filters, α, cross-section/Q, continuity check; raw-video orthorectification (Castor, Salmon) |
| 3 | Thu 22 | Tools + MCP server + rules planner + trace; degradation suite; agent eval v1. **AWS account live, COOL subscribed** |
| 4 | Fri 23 | AWS pipeline wired (S3/SQS/Graviton/DynamoDB/Step Functions/SNS); Claude-on-Bedrock planner; field capture if possible |
| 5 | Sat 24 | Qwen-in-OpenCV planner; COOL benchmark; web endpoint with live trace; pyORC baseline |
| 6 | Sun 25 | Full eval run, report, architecture diagram, judge one-pager, README |
| 7 | Mon 26 | Demo video (≤5 min), final checks from a clean clone, **submit by 6 PM PDT** (buffer) |

Cut order if behind: the local Qwen planner → pyORC baseline → STIV. Never cut the real-river
evaluation, the AWS wiring, or the human approval step.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| Reference coordinates are in ortho-frame pixels; our own ortho of the raw video differs | Use the provided ortho frames for scoring; use raw video + GCPs only to demonstrate the full path, and score after mapping via GCP world coordinates |
| Depth-averaged ADCP vs our surface velocity | Extrapolate the top bins to the surface (power/log law); report both |
| Low frame counts (Thalhofen 56, LaMorge 48 frames) | Report n; prefer the longer cases for the headline |
| Bedrock model access / AWS quota delays | Request access on day 3 at the latest; the rules planner always works without it |
| Qwen 0.5B makes poor decisions | Constrained schema plus tool guards; it's an edge fallback and we report its accuracy honestly |
| Slow data server | Range-pull only what we need; already running |
| Over-claiming | A "what this does not establish" section, as TremorCV has; no operational flood claims |

---

## 8. Needed from you

1. **AWS account** with credentials on this machine (`aws configure` / `aws login`). The CLI is
   installed and has no credentials. Then: request Bedrock Claude access, subscribe to the COOL
   Graviton4 listing (7-day trial covers the build window), and check the c8g vCPU quota.
2. **A stream, canal or drain you can film** for the field validation (30 min, phone, tape measure, 4 markers, an orange).
3. Team name, and whether it's solo (affects Devpost registration).
4. OK on the name **Gaugeless** (or pick another).

---

## 9. Submission checklist

- [ ] OpenCV 5 runtime reported by the app itself (`cv2.__version__`, build info)
- [ ] Meaningful AWS component running, with evidence in the video
- [ ] Live HTTPS endpoint, or an arranged screen-share
- [ ] Demo video ≤ 5 min: team, working app, architecture, OpenCV 5 usage, AWS, results
- [ ] Technical report: problem, users, architecture, OpenCV 5, AWS, evaluation, limitations, responsible use
- [ ] Public repo, pinned deps, build/deploy/test instructions, `make eval`
- [ ] Architecture diagram (input → OpenCV 5 → perception → agent → AWS → decision/action)
- [ ] Evaluation evidence incl. failure cases
- [ ] COOL section with provenance; Agentic section mapped to its rubric
- [ ] Devpost form submitted before 11:45 PM PDT Oct 26

### Day-2 late additions (Oct 19, before the usage limit)
- **Castor from raw video**, our own orthorectification (`gaugeless/ortho.py`, `prepare_castor.py`):
  - GCP camera solve with a k1 lens model: 2.6 px RMS, leave-one-out median 2.5 px (pinhole DLT: 16 px).
  - Water plane at the 3.772 m stage, one `cv2.remap` per frame (4 ms).
  - SIFT registration to the published grid (it's Y-mirrored), used to carry the reference positions across.
  - Agent result: points 15.6% vs ADCP depth-averaged, **Q 25.5 vs 29.6 m³/s (−13.8%)**.
  - Caveat: our scale and the published grid's scale differ by 3–6%.
- **Discharge is now validated at 5 sites:** La Morge −0.4%, Thalhofen +1.0%, Murg +3.4%, La Vence −10.2%, Castor −13.8%.
- **Qwen 1.5B in cv2.dnn** (same decision test as 0.5B): 1/6 correct with raw numbers, 3/6 with PROBLEM/ok labels, and it
  misses "refuse unseeded". Small local models can't plan. Claude/Bedrock is the planner; Qwen-in-OpenCV writes offline reports.
- Still running at cut-off: `results/degrade_v3.jsonl` (all fixes). Summarise it with `python -m gaugeless.agent.summarise results/degrade_v3.jsonl`.

**Next:**
1. Summarise degrade v3 into the README.
2. Run the Claude planner on all cases (planner comparison table).
3. Once the user OKs it: deploy AWS (`make deploy`), subscribe to COOL, run `bench` stock vs COOL.
4. Demo video and submission.

### Degradation suite v3 (Oct 20), final scoring
- 105 runs on 7 rivers.
- Damage-induced silent failures: **fixed 24, agent 0**.
- Agent vs fixed on the same points: 45 better, 56 equal (±2 points), 4 worse (Thalhofen flicker, +0.3–1.1 points).
- Escalations: 3, all at glare 0.3.
- Scoring bug fixed: the fixed-pipeline Q was compared against La Morge's reference for every sectioned site.
- Silent failure is now defined relative to the clean clip, so Tiber and GoPro (over 40% even when clean) aren't counted as damage.

### Planner comparison (Oct 20): rules vs Claude Sonnet 4.5 on Bedrock, same tools, 14 rivers (`results/agent_claude.jsonl`)
- Same outcome (publish/refuse) on 14/14: both refuse the two unseeded clips. Same Q on La Morge, Thalhofen,
  La Vence and Castor; Murg +8.6% (Claude added flow stabilisation) vs +3.4% (rules).
- Point MAPE:
  - Claude better on Bradano (19.5 vs 24.2), Thalhofen (22.0 vs 24.1) and Murg (16.7 vs 20.2).
  - Claude worse on Noce (21.8 vs 14.4: it added an unnecessary glare mask), Brenta (53 vs 44) and La Vence (18.7 vs 14.9).
- Claude takes 15–60 s per clip (LLM turns) vs 2–25 s for rules.
- **Gap found:** on Tiber, Claude once published without running sample_points. New guard rail: `publish` refuses
  until gauge points are sampled and sections computed (test added; 25 tests pass).
- Takeaway for the report: guard rails make an LLM planner safe, and it's as good as hand-written rules with zero
  site-specific code. Rules stay the deterministic floor, and the local Qwen model can't plan (1–3/6).

## AWS deployment (Oct 20) — live, us-east-1, stack `gaugeless`
- Account is on the AWS **free plan** ($27.66 credit left, expires Dec 2; account closes if credit runs out).
  → t4g.small Graviton worker (free-plan eligible), 20 GB disk, 4 GB swap, $20 gross-cost budget alert.
- COOL is **not possible** on this account (Marketplace needs a paid plan). Best Use of COOL is dropped unless a
  paid-plan account appears.
- Deploy fixes: SG egress needs VpcId (removed); worker needed s3:ListBucket for data sync; refs.ROOT was a
  hard-coded Mac path (now relative / GAUGELESS_DATA); worker skips the s3:TestEvent message.
- Cloud results (aarch64, OpenCV 5.0.0, rules planner, S3 → SQS → worker → DynamoDB):

| Job | Outcome | Time | Note |
|---|---|---|---|
| La Morge | published, Q 1.316 m³/s (−0.4%) | 54 s (1.13 s/frame) | same as laptop |
| Noce | published | 80 s | |
| Arrow unseeded | **refused** (no trackable surface) | 11 s | |
| La Morge, flood_q 1.0 | **escalated** → email → human **rejected** → recorded, no alert | 53 s | |
| La Morge, flood_q 1.0 | **escalated** → email → human **approved** → SNS alert sent → recorded | 53 s | |
