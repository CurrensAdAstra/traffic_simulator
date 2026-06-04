# Experimental Design — GPU-Accelerated Multi-Resolution Traffic Simulation

Design document for a paper built on this project's three engine families
(edge-macro / lane-CTM / mesoscopic) validated against SUMO microsimulation.
Goal: turn the existing exploratory results into reviewer-defensible claims.

---

## 1. Framing & contributions

**Recommended framing (sweet spot for our results):** a *systems-for-transportation*
paper — "GPU-accelerated multi-resolution traffic simulation, validated against
microsimulation at city scale." Targets **IEEE T-ITS** or **Transportation
Research Part C** (both reward speed *and* fidelity). Secondary venue if we lean
HPC: IPDPS / SC workshop.

**Claimed contributions (each maps to experiments below):**
- **C1.** A single framework spanning three fidelity levels on identical inputs,
  with a swap-one-flag dispatcher and a unified SUMO-validation harness. *(E1)*
- **C2.** GPU lane-CTM achieving ~10⁴× wall-clock speedup over SUMO while
  preserving macroscopic correlation; characterized scaling in network size and
  demand. *(E2)*
- **C3.** A principled calibration (free-flow scale + HCM movement capacity) with
  **held-out validation**, not in-sample tuning. *(E3, E4)*
- **C4.** A mesoscopic engine recovering per-vehicle travel-time distributions
  (a fidelity axis macro models cannot reach), with an honest accuracy ceiling
  analysis. *(E5, E6)*
- **C5.** Reproducible negative results delimiting where added model complexity
  does *not* help. *(E7)*

**Research questions:**
- RQ1 (speed–accuracy): Across engines, what is the Pareto frontier of
  SUMO-agreement vs wall time?
- RQ2 (scaling): How does GPU speedup scale with |edges| and |vehicles|, and
  where is the CPU/GPU crossover?
- RQ3 (transfer): Does a calibration learned on one network/demand transfer to
  unseen ones?
- RQ4 (attribution): Which model components (CTM vs LWR, turn-split, priority,
  HCM, FD-scale) drive the accuracy gains?
- RQ5 (meso fidelity): How well, and under what conditions, does the meso engine
  reproduce SUMO per-vehicle travel times?

---

## 2. Experimental factors (independent variables)

| Factor | Levels | Notes |
| --- | --- | --- |
| **Engine** | edge-LWR, lane-CTM, meso | + backend {CPU, GPU} for CTM/edge |
| **Network** | ≥4: Gangnam-4 + 3 standard open scenarios | see §6 — external validity |
| **Demand scale** | 0.25×, 0.5×, 1×, 2× base | spans free-flow → gridlock |
| **Calibration** | uncalibrated, calibrated(train), calibrated(test) | held-out |
| **SUMO seed** | 10 seeds | microsim is stochastic → CIs |

Controlled: same net.xml, same route file (depart-sorted), same sim horizon
(3600 s), same FD params (jam density 0.18 veh/m/lane), same hardware.

---

## 3. Metrics (dependent variables) — use traffic-standard, not just Pearson

**Edge-level (vs SUMO edgeData, SUMO-observed edges only):**
- **GEH statistic** on flow: report **% edges with GEH < 5** (the traffic-eng
  acceptance threshold) — primary flow metric, replaces ad-hoc Pearson.
- **RMSN** (RMS normalized) for speed and density.
- **Pearson r** and **Spearman ρ** for speed/density (ρ is the honest
  congestion-*ranking* metric).
- **Hotspot detection**: precision/recall@K and Spearman ρ over edge speeds —
  replaces the Jaccard we used (Jaccard conflates rank with set membership).

**Vehicle-level (meso only, vs SUMO tripinfo):**
- Travel-time **Pearson r** + **Spearman ρ** (matched by id).
- **MAPE** and **bias** of per-vehicle travel time.
- Distribution agreement: **KS statistic** + **Theil's U** on travel-time CDFs.
- **Completion-rate** agreement (arrived / loaded) — guards against the gridlock
  confound we found (compare only when both > ~85%).

**Performance:**
- Wall time (warm, median of 5 runs), and **throughput** = (|veh| × steps)/s and
  (|edges| × steps)/s — lets cross-network comparison.
- Speedup vs SUMO (same hardware, single-thread SUMO as the standard baseline).
- GPU: also report kernel time vs end-to-end (load + transfer + sim + write).

**Statistics:** every accuracy number is **mean ± 95% CI over the 10 SUMO
seeds**. Engine-vs-engine claims use **paired Wilcoxon** across edges/vehicles
(non-normal). State the test and n in captions.

---

## 4. Core experiments

### E1 — Speed–accuracy Pareto frontier (the money figure) → C1, RQ1
- Run every (engine × backend × calibration) on Gangnam-4 at demand 1×.
- Plot: x = wall time (log), y = accuracy (GEH<5% for flow, and a second panel
  for speed Spearman ρ). Each point = mean over seeds, error bars = 95% CI.
- **Expected story:** lane-CTM-GPU sits at the fast extreme (~0.1 s) on a high
  flow-correlation iso-line; meso sits at higher accuracy / higher cost; SUMO is
  the far-right reference at full fidelity. Pareto-dominated points (edge-LWR,
  uncalibrated) shown for context.
- Deliverable: **Figure 1** + Table of all points.

### E2 — Scaling study → C2, RQ2
Two sweeps, GPU and CPU:
- **Vehicles:** 10k, 25k, 50k, 100k, 250k, 500k, 1M (replicate/duplicate demand
  on Gangnam-4 to reach 1M). Fixed network. Report wall time + throughput.
- **Network size:** subgraphs of the full net (¼, ½, 1×) and a tiled/replicated
  2×, 4× to push |edges|. Fixed per-edge demand density.
- Identify **CPU↔GPU crossover** point and **GPU saturation** (when SMs fill).
- Report **strong** (fixed problem, this is single-GPU) and the kernel-vs-overhead
  breakdown (CUDA Graphs effect already measured: 1.36 s → 0.09 s).
- Deliverable: **Figure 2** (wall time vs size, CPU/GPU/SUMO) + throughput table.

### E3 — Held-out calibration → C3, RQ3
Defeats the "you tuned on the test set" critique (our biggest current weakness).
- **Split:** calibrate (vmax-scale, major-left, minor factors) on a **training
  set** = {Gangnam-4 @ demand 0.5×} (or network A); evaluate frozen params on
  **test set** = {other demand levels} and {other networks}.
- Calibration method: grid or Nelder–Mead minimizing edge-flow RMSN on train
  only; report the chosen params once.
- **Report train vs test accuracy gap.** A small gap is the headline of C3.
- Deliverable: Table — accuracy(train) vs accuracy(test) per network/demand.

### E4 — Component ablation → C4, RQ4
Incrementally enable, measure ΔGEH<5% and Δspeed-ρ at each step on the test set:
1. lane-LWR baseline → 2. + CTM sending/receiving → 3. + priority/yield →
4. + route-derived turn split → 5. + HCM movement capacity → 6. + FD vmax-scale.
- We already have single-config numbers; this needs the clean per-seed runs.
- Deliverable: **Table** (waterfall of contributions) — shows turn-split and
  FD-scale are the dominant levers (consistent with exploratory findings).

### E5 — Validation rigor across demand & time → C4, RQ5
- For each demand level (0.25–2×), run SUMO ×10 seeds → ground truth with CIs.
- Report edge metrics **per 5-min interval** (not just the 1-hour average) to
  show temporal tracking, plus the aggregate.
- **Critical control:** only compute travel-time agreement where both SUMO and
  meso completion > 85% (the un-gridlocked regime; our 20k clean case showed
  r 0.31→0.45 once this confound was removed).
- Deliverable: **Figure 3** (accuracy vs demand level, per engine) + interval plots.

### E6 — Mesoscopic per-vehicle fidelity → C4
- Clean-regime travel-time validation (demand levels where both finish):
  Pearson/Spearman, MAPE, KS, bias.
- **Congestion-dependent junction delay** study: compare flat `--junction-delay`
  (shown to trade bias for correlation) vs a demand-scaled / queue-length-scaled
  delay. Hypothesis: congestion-dependent delay closes bias *without* the
  correlation loss the flat delay caused.
- Deliverable: travel-time CDF overlay (meso vs SUMO) + the junction-delay
  ablation table.

### E7 — Negative results → C5
Document, with the same rigor (CIs), the three explored dead-ends:
aggregate junction-capacity cap, multi-cell spatial refinement, stochastic flow
noise — each shown not to move the time-averaged metrics, with the diagnosis
(time-averaging insensitivity; uniform vs spatially-varying effects).
- Deliverable: short table + one paragraph each. Strengthens credibility.

---

## 5. What must be BUILT for paper-grade rigor (gaps vs current code)

Current code does single-run point comparisons. Paper needs:
1. **Multi-seed SUMO runner** — wrap SUMO with `--seed`, N=10, aggregate
   edgeData+tripinfo to mean±CI. (extend `run_compare_sumo.py`).
2. **GEH + RMSN + Spearman + KS** metrics in the compare harness (currently only
   Pearson/MAE/RMSE/Jaccard). Add `--metrics geh,rmsn,spearman,ks`.
3. **Scaling harness** — parametric demand replication + subnetwork extraction;
   automated wall-time/throughput logging (warm runs, median of 5).
4. **Calibration driver** — Nelder–Mead/grid over (vmax-scale, ml, minor) on a
   train split, freeze, evaluate on test. ~80 lines.
5. **Demand-scaling tool** — subsample/replicate the route file to 0.25×–2× (and
   to 1M for scaling). Trivial extension of the 20k subset script already written.
6. **Per-interval edgeData** — SUMO `<edgeData period="300">` + meso/CTM
   time-binned output (meso already tracks time; CTM needs interval dumps).
7. **Plotting** — Pareto (Fig 1), scaling (Fig 2), accuracy-vs-demand (Fig 3),
   travel-time CDF, kept out of the engines (separate `paper/plots.py`).

## 6. External validity — additional networks (acquire)

One network = anecdote; reviewers require ≥3. Use **standard open SUMO
scenarios** (already in SUMO-net format, citable, reproducible):
- **LuST** (Luxembourg SUMO Traffic) — full-day city scenario, ~900 edges core.
- **MoST / Monaco** — mixed urban.
- **TAPASCologne** — large, ~1M trips/day → good for the scaling claim too.
- **Gangnam-4** (ours) — the home network.
Run the *entire* pipeline on each; calibration trained on one, tested on others
(E3). This single addition is the biggest lift in paper acceptability.

## 7. Reproducibility (reviewers increasingly require)

- Release the Docker image (`gangnam4-cuda-sumo`, pinned SUMO 1.27, cupy 14),
  all scripts, fixed seeds, and the calibration params.
- Report exact hardware (RTX 3090, 24 GB; CPU model; CUDA 13 driver).
- A `make reproduce` that regenerates every table/figure from raw runs.
- Archive nets/routes + SUMO configs (DOI via Zenodo).

## 8. Threats to validity (write this section honestly)

- **Calibration transfer** — mitigated by E3 held-out; report the residual gap.
- **SUMO as "ground truth"** — SUMO is itself a model; frame agreement as
  *cross-model consistency*, not absolute truth. Cite SUMO's own calibration.
- **Gridlock/teleport confound** — discovered here; mitigated by the
  completion-rate gate (E5) and demand sweep.
- **Single GPU / single vendor** — scope the speedup claim to one A-class GPU;
  note CPU baseline is multithreaded NumPy (state thread count).
- **FD/saturation-flow priors** — report sensitivity (the sat-flow & min-speed
  sweeps already show low sensitivity of edge metrics).

---

## 9. Minimal vs full paper (scoping)

- **Workshop / short paper:** E1 + E2 + E4 on Gangnam-4 only (speed-accuracy +
  scaling + ablation). Mostly runnable with current code + metrics upgrade.
- **Full journal paper:** all of E1–E7 across ≥4 networks with multi-seed CIs and
  held-out calibration. Needs §5 build-out + §6 networks.

Recommended target: full journal (T-ITS / TR-C). The multi-resolution + GPU +
held-out-validation combination is a genuine, defensible contribution.
