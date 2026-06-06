# Benchmark Results — 4(+1) Simulator Performance Comparison

Performance comparison of traffic-simulation engines on the Gangnam-4 network
(8,469 edges / 17,810 lanes), validated against SUMO. Hardware: **RTX 3090
(24 GB), CUDA 13 driver, cupy 14**; SUMO 1.27 (single-thread) and CPU engines on
the same host. Reproduce with `gangnam4_cuda/run_benchmark.py`.

`sim_wall` = simulation-loop wall (excludes load/parse), the fair algorithmic
metric. `RTF` = real-time factor = sim_time / sim_wall. Each engine's accuracy
(vs SUMO) is reported alongside speed — fast-but-wrong is not useful.

## Table 1 — Systems @ Gangnam-4, 100k vehicles, 3600 s sim

| System | sim_wall | RTF | Speedup vs SUMO | SUMO-agreement (flow) |
| --- | --- | --- | --- | --- |
| **SUMO** (microscopic) | 1685 s | 2.1× | 1× | reference |
| **Macro** CPU (lane-CTM) | 7.4 s | 487× | **228×** | r≈0.69 |
| **Meso** CPU (vehicle queue) | 23.4 s | 154× | 72× | r 0.84, GEH<5 98%¹ |
| **Meso** GPU — naive port | 20.7 s | 174× | 81× | = meso (bit-exact) |
| **Meso** GPU — **branch-free + CUDA graph** | **0.12 s** | **30,000×** | **~14,000×** | flow r 0.99 vs meso² |

¹ at moderate demand (clean regime); ² atomic-ticket approximation, ~1–2% vs exact-FIFO meso.
dt: meso engines 1.0 s (3600 steps); macro-CTM 0.5 s (7200 steps).

## Table 2 — Scaling: sim_wall (s) vs vehicle count

| Vehicles | Macro CPU (CTM) | Meso CPU | Meso GPU (naive) | **Meso GPU (graph)** | SUMO |
| --- | --- | --- | --- | --- | --- |
| 10k | 3.7* | 1.6 | 7.7 | — | — |
| 100k | 3.7* | 23.4 | 20.7 | **0.12** | 1685 |
| 500k | 3.7* | 136 | 23.0 | **0.27** | (intractable) |
| 1M | 3.7* | ~270 | 25.6 | **0.45** | (intractable) |

\*Macro-CTM is **O(edges), independent of vehicle count** (demand enters once at
boot); measured 3.66–3.70 s flat across 10k–500k at dt=1.0. Meso/SUMO are
O(vehicles). The redesigned Meso-GPU stays near-flat (0.12→0.45 s for 100k→1M).

## The headline finding — breaking the mesoscopic-GPU "10× ceiling"

A naive vectorized GPU port of a time-stepped mesoscopic model (the common
2019-era approach, and our first `meso_gpu.py`) is capped near **10×** over CPU.
Profiling showed why: the cost is **host–device synchronization + many small
kernel launches** (`flatnonzero` / `.any()` / `.size` / scalar reads every
step), **not** the per-step sort (removing the sort gave only 1.1×).

Redesigning the step to be **sync-free, branch-free, and CUDA-Graph-captured**
(`meso_gpu_graph.py`) removed that floor:

| Lever (isolated) | 100k sim_wall | gain |
| --- | --- | --- |
| naive meso-GPU | 20.7 s | 1× |
| + branch-free (no host sync / flatnonzero; all-V fixed kernels, atomic tickets) | 0.31 s | **67×** |
| + CUDA graph capture (kill launch overhead) | **0.12 s** | **172×** total |

This is the same pattern that took the macro-CTM GPU engine from 1.36 s → 0.09 s
(15×). Net: meso-GPU now reaches **57–172× over the naive port**, **~600× over
meso-CPU at 1M**, and lands in the macro-CTM performance regime (CTM @1M ≈
0.09 s; meso-graph @1M = 0.45 s) — while retaining vehicle-level fidelity
(per-vehicle travel times, flow agreement r≈0.99 vs exact-FIFO).

**Conclusion:** beyond ~10× is not a hardware limit; it requires changing the
algorithm's data-flow structure (eliminate global sync/sort, fix kernel shapes,
capture as a graph), not just porting the existing loop to the GPU.

## Speed–accuracy positioning (which engine when)

- **Macro-CTM (CPU/GPU):** fastest at scale (demand-independent), best at
  *ranking* congestion; coarsest accuracy (flow r≈0.69, no per-vehicle data).
- **Meso-GPU (graph):** vehicle-level fidelity (per-vehicle travel times) at
  near-macro speed — the new sweet spot for large, high-demand scenarios.
- **SUMO (micro):** highest fidelity (continuous x,y, car-following, signals);
  intractable at city-scale high demand (28 min for 100k, RTF ≈ 2×).

## Per-vehicle position trackability (verification axis)

| Engine | Single-vehicle position | Granularity |
| --- | --- | --- |
| SUMO | yes (fcd) | continuous x,y + lane + pos |
| Macro-CTM | **no** | density field only (no vehicles) |
| Meso (CPU/GPU) | yes (`--track-vehicle`) | edge id + meters-along-edge |
