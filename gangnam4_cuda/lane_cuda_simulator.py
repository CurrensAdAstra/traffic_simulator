#!/usr/bin/env python3
"""
CUDA 기반 차선(lane)-병렬 교통 시뮬레이터.

- 도로 1개 차선 = GPU 스레드 1개
- edge 단위 CUDA 엔진(`cuda_edge_simulator.py`)의 lane 단위 확장판
- CPU 버전(`lane_cpu_simulator_mt.py`)과 동일한 2단계 모델:
    (1) 종방향 LWR 갱신 커널, (2) 회전 수요 기반 횡방향 차선변경 커널
- `lane_common`의 로더/수요 빌더를 공유

실행은 GPU/CUDA 환경(예: Dockerfile 이미지 `gangnam4-cuda-sumo:latest`)에서 한다.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from lane_common import (
    LaneNet,
    aggregate_to_edges,
    build_demand_and_target,
    load_lane_net,
    log,
    fail,
)


def build_kernels(cp):
    long_code = r'''
    extern "C" __global__
    void long_kernel(
        const int n,
        const int* in_ptr, const int* in_lanes, const float* in_w,
        const float* rho, const float* flow, const float* vmax,
        const float rho_jam, const float* length, const float dt,
        const float* source_demand, float* rho_long
    ) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= n) return;
        int s = in_ptr[i];
        int e = in_ptr[i + 1];
        float inflow = source_demand[i];
        if (e > s) {
            inflow = 0.0f;
            for (int k = s; k < e; ++k) inflow += flow[in_lanes[k]] * in_w[k];
        }
        float v = vmax[i] * (1.0f - rho[i] / rho_jam);
        if (v < 0.0f) v = 0.0f;
        float outflow = rho[i] * v;
        float r = rho[i] + dt * (inflow - outflow) / fmaxf(length[i], 1.0f);
        if (r < 0.0f) r = 0.0f;
        rho_long[i] = r;
    }
    '''

    lat_code = r'''
    extern "C" __global__
    void lat_kernel(
        const int n,
        const int* lat_ptr, const int* lat_nb,
        const float* rho_long, const float* target_share, const float* edge_rho,
        const int* lane_edge, const float* vmax, const float rho_jam,
        const float dt, const float k_lc,
        float* rho_next, float* speed, float* flow_next
    ) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= n) return;
        int ei = lane_edge[i];
        float er = edge_rho[ei];
        float phi_i = rho_long[i] - target_share[i] * er;
        int s = lat_ptr[i];
        int e = lat_ptr[i + 1];
        float lat = 0.0f;
        for (int k = s; k < e; ++k) {
            int j = lat_nb[k];
            float phi_j = rho_long[j] - target_share[j] * er;
            lat += (phi_j - phi_i);
        }
        float r = rho_long[i] + dt * k_lc * lat;
        if (r < 0.0f) r = 0.0f;
        if (r > rho_jam) r = rho_jam;
        float v = vmax[i] * (1.0f - r / rho_jam);
        if (v < 0.0f) v = 0.0f;
        rho_next[i] = r;
        speed[i] = v;
        flow_next[i] = r * v;
    }
    '''
    return cp.RawKernel(long_code, "long_kernel"), cp.RawKernel(lat_code, "lat_kernel")


def run_sim(args) -> None:
    try:
        import cupy as cp  # type: ignore
        import cupyx  # type: ignore
    except Exception as e:
        fail(f"cupy import 실패: {e}")

    net = load_lane_net(Path(args.net_file))
    n = net.n_lanes
    threads = int(args.threads_per_block)
    blocks = (n + threads - 1) // threads
    log(f"CUDA(lane) 설정: lanes={n}, edges={net.n_edges}, threads/block={threads}, blocks={blocks}")

    rho_jam = np.float32(args.jam_density_per_lane)

    source_demand_np, target_share_np, veh_n = build_demand_and_target(
        net, Path(args.net_file), Path(args.route_file) if args.route_file else None,
        args.sim_duration, args.source_demand,
    )
    log(f"수요/목표분배 준비 완료: vehicles={veh_n}, lane-change-rate={args.lane_change_rate}")

    rng = np.random.default_rng(args.seed)
    rho0 = (args.init_density * rng.uniform(0.7, 1.3, size=n)).astype(np.float32)
    rho0 = np.clip(rho0, 0.0, float(rho_jam) * 0.95)

    # 디바이스 배열
    rho = cp.asarray(rho0)
    rho_long = cp.empty_like(rho)
    rho_next = cp.empty_like(rho)
    speed = cp.zeros_like(rho)
    flow = cp.zeros_like(rho)
    flow_next = cp.zeros_like(rho)

    vmax = cp.asarray(net.vmax_mps)
    length = cp.asarray(net.length_m)
    in_ptr = cp.asarray(net.in_ptr)
    in_lanes = cp.asarray(net.in_lanes)
    in_w = cp.asarray(net.in_w)
    lat_ptr = cp.asarray(net.lat_ptr)
    lat_nb = cp.asarray(net.lat_neighbors)
    lane_edge = cp.asarray(net.lane_edge)
    source_demand = cp.asarray(source_demand_np)
    target_share = cp.asarray(target_share_np)
    edge_rho = cp.zeros(net.n_edges, dtype=cp.float32)

    long_kernel, lat_kernel = build_kernels(cp)

    def run_step() -> None:
        long_kernel(
            (blocks,), (threads,),
            (np.int32(n), in_ptr, in_lanes, in_w, rho, flow, vmax, rho_jam,
             length, np.float32(args.dt), source_demand, rho_long),
        )
        edge_rho.fill(0)
        cupyx.scatter_add(edge_rho, lane_edge, rho_long)
        lat_kernel(
            (blocks,), (threads,),
            (np.int32(n), lat_ptr, lat_nb, rho_long, target_share, edge_rho,
             lane_edge, vmax, rho_jam, np.float32(args.dt),
             np.float32(args.lane_change_rate), rho_next, speed, flow_next),
        )

    # 초기 1스텝
    run_step()
    rho, rho_next = rho_next, rho
    flow, flow_next = flow_next, flow

    t0 = time.perf_counter()
    for step in range(args.steps):
        run_step()
        rho, rho_next = rho_next, rho
        flow, flow_next = flow_next, flow

        if (step + 1) % args.log_interval == 0:
            log(
                f"step={step+1}/{args.steps} mean_speed={float(cp.mean(speed).get()):.3f} m/s "
                f"mean_density={float(cp.mean(rho).get()):.6f} veh/m"
            )

    cp.cuda.runtime.deviceSynchronize()
    elapsed = time.perf_counter() - t0
    log(f"CUDA(lane) 시뮬레이션 완료: {elapsed:.3f}s ({args.steps} steps)")

    write_outputs(args, net, cp.asnumpy(rho), cp.asnumpy(speed), cp.asnumpy(flow))


def write_outputs(args, net: LaneNet, rho: np.ndarray, speed: np.ndarray, flow: np.ndarray) -> None:
    speed_ratio = speed / np.maximum(net.vmax_mps, 1e-6)
    travel_time = net.length_m / np.maximum(speed, 0.1)

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "lane_id", "edge_id", "lane_index", "length_m", "vmax_mps",
            "density_veh_per_m", "speed_mps", "flow_veh_per_s", "speed_ratio", "travel_time_s",
        ])
        for i in range(net.n_lanes):
            w.writerow([
                net.lane_ids[i],
                net.edge_ids[int(net.lane_edge[i])],
                int(net.lane_local[i]),
                float(net.length_m[i]),
                float(net.vmax_mps[i]),
                float(rho[i]),
                float(speed[i]),
                float(flow[i]),
                float(speed_ratio[i]),
                float(travel_time[i]),
            ])
    log(f"lane 결과 저장: {out}")

    if args.edge_output_csv:
        agg = aggregate_to_edges(net, rho, speed, flow)
        eout = Path(args.edge_output_csv)
        eout.parent.mkdir(parents=True, exist_ok=True)
        with eout.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["edge_id", "lanes", "density_veh_per_m", "speed_mps", "flow_veh_per_s"])
            for eid, (dens, msp, fl, lanes) in agg.items():
                w.writerow([eid, lanes, dens, msp, fl])
        log(f"edge 집계 저장: {eout}")

    worst = np.argsort(speed_ratio)[: args.topk]
    log("혼잡 상위 lane")
    for idx in worst:
        log(
            f"  lane={net.lane_ids[idx]} speed={speed[idx]:.3f}/{net.vmax_mps[idx]:.3f} "
            f"ratio={speed_ratio[idx]:.3f} density={rho[idx]:.5f}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Lane-per-thread CUDA traffic simulator")
    p.add_argument("--net-file", default="./map_import/gangnam4_generated.net.xml")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--jam-density-per-lane", type=float, default=0.18, help="veh/m/lane")
    p.add_argument("--init-density", type=float, default=0.03, help="veh/m")
    p.add_argument("--source-demand", type=float, default=0.02, help="veh/s")
    p.add_argument("--route-file", default="./map_import/gangnam4_generated.gpu_compatible.rou.xml")
    p.add_argument("--sim-duration", type=float, default=86400.0)
    p.add_argument("--lane-change-rate", type=float, default=0.5, help="횡방향 완화 계수 k_lc")
    p.add_argument("--threads-per-block", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=200)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--output-csv", default="./gangnam4_cuda/results/lane_state_cuda.csv")
    p.add_argument("--edge-output-csv", default="./gangnam4_cuda/results/lane_state_cuda.edge.csv")
    args = p.parse_args()
    run_sim(args)


if __name__ == "__main__":
    main()
