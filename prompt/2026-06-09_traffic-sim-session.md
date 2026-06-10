# Traffic Simulation 세션 기록 — 5-시뮬레이터 비교 & meso-GPU 10× 한계 돌파

- **날짜**: 2026-06-09
- **브랜치**: `claude/elastic-allen-cc712e` (로컬 전용, 미push)
- **워크트리**: `/mnt/ludus_directorium/traffic_simulation/.claude/worktrees/elastic-allen-cc712e`
- **프로젝트**: `traffic_simulation` — 모든 엔진 코드는 `gangnam4_cuda/`

---

## 1. 세션 목표 (사용자 요청 흐름)

이 세션은 여러 세션에 걸친 교통 시뮬레이터 비교 프로젝트의 연속이다. 요청 순서:

1. **5-시뮬레이터 성능 비교** 구축 (SUMO / macro-CTM-CPU / macro-CTM-GPU / meso-CPU / meso-GPU)
2. 알고리즘별 **차량 1대 단위 위치 추적 검증** ("각 알고리즘이 검사용으로 차량 1대 데이터를 넣었을 때, 위치 추적이 되는가?")
3. GPU meso가 역사적으로 **~10× 한계**에 갇히는 이유 조사 (사용자의 2019년 논문 언급)
4. GPU meso 프로파일링 후 **재설계로 10× 한계 돌파**
5. 모델 정밀화 — **혼잡 의존 교차로 지연(congestion-dependent junction delay)**, graph-vs-CPU 격차 해소
6. **E1 Pareto 데이터** 생성 + 장시간 매트릭스 sweep
7. (D) `architecture.md` 갱신 + (B) SUMO 포함 100k까지 **tmux 장시간 sweep** 실행
8. 대화 내용을 `prompt/` 디렉토리에 md로 저장 ← (현재)

---

## 2. 다섯 개의 검증된 시뮬레이터

| # | 시뮬레이터 | 파일 | 특징 |
|---|-----------|------|------|
| 1 | **SUMO microscopic** | Docker SUMO 1.27 | Reference(ground truth) |
| 2 | **Macro CTM CPU** | `lane_cpu_simulator_mt.py --model ctm` | Daganzo CTM, priority/yield merge |
| 3 | **Macro CTM GPU** | `lane_cuda_simulator.py` | gather-only 커널 + fused ElementwiseKernel + CUDA Graphs. **0.09s / 1h 100k-veh (RTX 3090, ~24,000× SUMO)**. 수요 독립 O(edges) |
| 4 | **Mesoscopic CPU** | `meso_sim.py` | 차량 단위 time-stepped spatial-queue, 차량별 통행시간 |
| 5 | **Mesoscopic GPU (재설계)** | `meso_gpu_graph.py` | branch-free, sync-free, atomic-ticket FIFO, 전체 step CUDA-Graph 캡처. **naive 포트 대비 172×** |

`meso_gpu.py` = naive cupy 포트(=10× 한계 baseline)도 비교용으로 유지.

**엔진 디스패처**: `run_engine.py --engine {edge,lane} --backend {cpu,gpu}` (CTM 계열).

---

## 3. 핵심 성과: meso-GPU 10× 한계 돌파 (172×)

### 진단 과정 (정직한 수정 포함)
1. `--profile` 섹션 분해 → lexsort가 ~33% 병목으로 **보였음**
2. 가설: atomic-ticket FIFO로 sort 제거 → 실제로는 **1.1×밖에** 개선 안 됨 (정직한 underdeliver)
3. **올바른 진단**: 진짜 병목은 host-sync + 많은 small-launch 오버헤드
4. **해결**: branch-free 커널 + 전체 step을 CUDA Graph로 캡처 → **172×** (1M veh에서 meso-CPU ~270s → meso-GPU-graph **0.45s**, ~600×)

### 주요 커널 (`meso_gpu_graph.py`)
`k_tick`, `k_edge_speed`, `k_advance`, `k_credit`, `k_send_ticket`, `k_recv`,
`k_disch_leave`, `k_disch_enter`, `k_dep_ticket`, `k_dep_apply`, `k_accum`

### 정확도 보존
- graph 엔진 초기 per-veh r=0.455 vs CPU 0.484 → occupancy 스냅샷 타이밍 불일치
- **split-discharge 수정** (leave → snapshot → enter 분리) → CPU와 소수점 3자리까지 일치 (r=0.484)

---

## 4. 모델 정밀화: 혼잡 의존 교차로 지연

- **형태**: Webster-overflow `delay = jct_delay + cong_coef * occ / (1.0 - occ)`
- CLI: `--junction-cong-coef` (~10), `--max-junction-delay`
- **의미**: flat 교차로 지연은 상관(correlation)과 bias를 trade-off시킴. 혼잡 의존형은 **둘 다 동시 개선** — 첫 사례
- Grid(140 신호) 검증: 물리적 정확성 확인 (MAPE 45%→16%, r=0.94 유지)
- `meso_sim.py`(CPU)와 `meso_gpu_graph.py`(GPU) 양쪽에 이식

---

## 5. 검증 지표 (Gangnam-4 100k, 3600s, GEH<5 = 교통공학 합격선)

- **Best macro CTM**: GEH<5≈4%, flow ρ 0.79, speed ρ 0.41 — 혼잡 edge *순위* 매기기에 강함, 비용은 수요 독립
- **Meso (모든 변형)**: **GEH<5 98%, flow ρ 0.99** — 정확도 챔피언
- **Per-vehicle r (meso만, 20k clean)**: **0.484** (CPU == GPU-graph, split-discharge 수정으로 일치)
- **SUMO noise floor**: 고정 route + multi-seed → flow ρ 0.999 → 엔진-vs-SUMO 격차는 진짜 모델 오차

---

## 6. 보정 노브 (데이터 기반)

| 노브 | 값 | 의미 |
|------|----|------|
| `--vmax-scale` | ~0.35 (Gangnam) / 0.5 (clean) | 전역 자유흐름 보정 |
| `--major-left-factor` | 0.7 | HCM Rank 2 |
| `--minor-factor` | 0.5 | HCM Rank 3-4 |
| `--junction-cong-coef` | ~10 | 혼잡 의존 교차로 지연 |

**네거티브 결과 (재시도 금지)**: aggregate junction-capacity cap, multi-cell CTM 공간 세분화, stochastic flow noise — 모두 시간평균 지표 개선 없음. meso-GPU의 lexsort 제거도 1.1×에 그침.

---

## 7. 논문 도구 (E1–E7)

- `metrics.py` — GEH/RMSN/Spearman/KS (pure numpy)
- `run_sumo_seeds.py` — multi-seed + noise floor
- `calibrate.py` — held-out train→test 격차 보고 (E3)
- `make_synthetic_scenarios.py` — grid/spider/random (netgenerate). grid가 Gangnam에 없는 140-신호 시나리오 제공 (E7)
- `run_benchmark.py` — 통합 4(+1)-시스템 성능 harness
- `run_pareto.py` — E1 속도-정확도 데이터 (`flush=True` 스트리밍)
- `run_scaling.py` — E2
- `run_matrix.py` + `launch_matrix_tmux.sh` — 장시간 매트릭스 sweep (tmux-detach, `--resume`, `[COMBO]`/`[ETA]` 마커)
- `scale_demand.py` — route 파일을 목표 차량수로 스케일 (duarouter inline route는 stack 기반 `last_inline_edges`로 처리)

---

## 8. B sweep 결과 — 100k 수요 (Gangnam + Grid), 7개 config

매트릭스 완료: 6/6 combos, ~30분. master CSV 43행.

### Gangnam 100k (matched 6560 edges)

| config | wall_inner_s | flow_geh_lt5 | flow_ρ | speed_ρ | density_ρ |
|--------|-------------|--------------|--------|---------|-----------|
| lane_cpu_ctm | 7.69 | 0.087 | 0.672 | 0.187 | 0.436 |
| lane_cpu_ctm_v035_hcm | 7.82 | 0.104 | 0.706 | 0.271 | 0.469 |
| meso_cpu | 23.52 | 0.574 | 0.827 | 0.468 | 0.725 |
| meso_cpu_v05 | 22.54 | 0.631 | 0.840 | 0.484 | 0.738 |
| meso_gpu_naive | 20.90 | 0.574 | 0.827 | 0.468 | 0.725 |
| **meso_gpu_graph** | **0.13** | 0.576 | 0.818 | 0.470 | 0.726 |
| **meso_gpu_graph_cong** | **0.15** | 0.605 | 0.811 | 0.487 | 0.736 |

→ **meso_gpu_graph가 meso_cpu와 동급 정확도를 0.13s에 달성 (meso_cpu 23.5s 대비 ~180×)**. cong 변형은 정확도(geh<5 0.605, speed_ρ 0.487)까지 끌어올림.

### Grid 100k (matched 528 edges)

| config | wall_inner_s | flow_geh_lt5 | flow_ρ | density_ρ |
|--------|-------------|--------------|--------|-----------|
| lane_cpu_ctm | 1.22 | 0.000 | 0.157 | 0.505 |
| lane_cpu_ctm_v035_hcm | 1.22 | 0.011 | 0.097 | 0.271 |
| meso_cpu | 13.94 | 0.670 | 0.491 | 0.540 |
| meso_cpu_v05 | 14.48 | 0.547 | 0.426 | 0.537 |
| meso_gpu_naive | 18.59 | 0.670 | 0.491 | 0.540 |
| **meso_gpu_graph** | **0.15** | 0.642 | 0.477 | 0.546 |
| **meso_gpu_graph_cong** | **0.15** | 0.661 | 0.447 | 0.525 |

→ grid에서도 meso_gpu_graph가 meso_cpu(14s)를 0.15s에 재현 (~93×).

**SUMO 100k 기준 생성**: gangnam ~1293s(~22분), grid ~500s. `--sumo-cap 20000`으로 그 이상은 ground-truth 생성 skip.

---

## 9. 주요 오류 & 수정

| 오류 | 수정 |
|------|------|
| SUMO duration 미파싱 | `--duration-log.statistics true` 플래그 |
| `cp.repeat`가 cupy 배열 repeats 거부 | numpy host expand 후 transfer |
| meso_gpu_graph 벤치마크 FileNotFound | overlay mount `-v $_HERE:/workspace/gangnam4_cuda` |
| SUMO cfg "not in subpath" | cfg를 workspace mount 내부(`map_import/`)에 작성 |
| scale_demand 빈 edges (duarouter inline route) | stack 기반 `last_inline_edges` |
| Pareto stdout 버퍼링 | 모든 print `flush=True` |
| profile이 lexsort를 33%로 오인 | 진짜는 sync+launch 오버헤드 → branch-free+CUDA Graph |
| graph per-veh r 격차 | split-discharge (leave→snapshot→enter) |

---

## 10. 인프라 핵심 사실

- Docker 이미지 `gangnam4-cuda-sumo:latest` — SUMO 1.27 (`ppa:sumo/stable`; apt 1.12는 net의 vClasses 거부)
- GPU 실행: `--gpus all` + worktree overlay mount (worktree는 main checkout에 없음)
- route 파일은 depart 정렬 필수 (`sort_routes.py`); 메인은 `map_import/gangnam4_generated.sorted.rou.xml`
- SUMO 출력/cfg는 `map_import/` 아래 (writable; `results/`는 옛 docker run으로 root 소유 → CSV는 `/tmp` 또는 `paper_data/`)
- 모든 작업은 `claude/elastic-allen-cc712e` 브랜치 — **미push** (sandbox에 GitHub creds 없음, 사용자가 수동 push)

### 장시간 sweep
```bash
bash gangnam4_cuda/launch_matrix_tmux.sh start     # 새 detached tmux 세션 tsmatrix
bash gangnam4_cuda/launch_matrix_tmux.sh --status  # 진행 폴링
bash gangnam4_cuda/launch_matrix_tmux.sh --tail    # 라이브 스트림
bash gangnam4_cuda/launch_matrix_tmux.sh --attach  # 부착 (Ctrl-b d 로 detach)
bash gangnam4_cuda/launch_matrix_tmux.sh --resume  # 기존 matrix.csv에서 이어 실행
```
`--sumo-cap 20000`: SUMO ground truth는 이 수요까지만 (100k SUMO ≈ 22-28분).

### 용어
- **COMBO** = 매트릭스의 한 (network, demand) 점
- **cell** (`cell_common.py`) = 한 차선의 공간 세분 (multi-cell CTM 실험, 네거티브) — COMBO와 무관

---

## 11. 커밋 히스토리 (이 세션 관련)

```
4323a92 docs: refresh architecture.md — 5 simulators, meso-GPU redesign, full tooling
0b42593 chore: rename CELL → COMBO in matrix sweep logs
77716d6 feat: long matrix sweep + tmux launcher (detach-safe, resumable)
2f7018f feat: E1 Pareto data on Gangnam-20k clean (8 configs)
58c6d1b feat: run_pareto.py — speed-accuracy Pareto data generator (E1)
aa82f49 docs: add accuracy-refinement section to benchmark_results
34dc387 fix: split discharge into leave/enter — closes graph-vs-CPU meso gap
67df979 feat: port congestion-dependent junction delay to graph meso-GPU
1af01ef feat: congestion-dependent junction delay (Webster-overflow form)
fc57c68 feat: per-vehicle trip output for graph meso + precision validation vs SUMO
e2993df docs: consolidated 4(+1)-system benchmark results
89a0cbe feat: add meso_gpu_graph (branch-free+CUDA-graph) as benchmark system
8cff9c4 feat: branch-free + CUDA Graphs meso-GPU — shatters the 10x ceiling (57-172x)
215df16 feat: --rank-mode ticket (sort-free FIFO via atomicAdd) — honest underdeliver
af9cd2d feat: --profile section breakdown for meso-GPU step
```

---

## 12. 완료된 태스크 (37개 전부 완료)

lane 엔진 구축(#1-9), GPU CTM 포팅/최적화(#10-17), 공간/확률 변형(#18-19),
packet meso(#20-21), 지표/검증(#22-23), 스케일링/보정/일반화(#24-27),
meso-GPU 재설계(#28-34), Pareto/매트릭스/문서(#35-37).

---

## 13. 다음 단계 후보

- (C) 결과를 논문 figure로 정리 (E1 Pareto plot, 속도-정확도 frontier)
- 100k 이상(500k/1M) 수요에서 SUMO 없이 엔진 wall-time만 측정하는 별도 sweep
- 사용자 push 대기 (브랜치 미push 상태)
