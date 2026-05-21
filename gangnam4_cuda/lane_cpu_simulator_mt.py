#!/usr/bin/env python3
"""
CPU 멀티스레딩 차선(lane) 단위 교통 시뮬레이터.

- edge 단위 엔진(`cpu_edge_simulator_mt.py`)의 lane 단위 확장판
- GPU 버전(`lane_cuda_simulator.py`)과 동일한 모델/입출력
- 각 스텝은 2단계: (1) 종방향 LWR 갱신, (2) 회전 수요 기반 횡방향 차선변경
- `lane_common`의 로더/수요 빌더를 공유
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from lane_common import (
    LaneNet,
    aggregate_to_edges,
    build_demand_and_target,
    load_lane_net,
    log,
)


def longitudinal_chunk(
    start: int,
    end: int,
    in_ptr: np.ndarray,
    in_lanes: np.ndarray,
    in_w: np.ndarray,
    rho: np.ndarray,
    flow: np.ndarray,
    vmax: np.ndarray,
    rho_jam: float,
    length: np.ndarray,
    dt: float,
    source_demand: np.ndarray,
    rho_long: np.ndarray,
) -> None:
    for i in range(start, end):
        s = int(in_ptr[i])
        e = int(in_ptr[i + 1])
        if e > s:
            inflow = float(np.dot(flow[in_lanes[s:e]], in_w[s:e]))
        else:
            inflow = float(source_demand[i])

        v = max(float(vmax[i]) * (1.0 - float(rho[i]) / rho_jam), 0.0)
        outflow = float(rho[i]) * v
        r = float(rho[i]) + dt * (inflow - outflow) / max(float(length[i]), 1.0)
        if r < 0.0:
            r = 0.0
        rho_long[i] = r


def lateral_chunk(
    start: int,
    end: int,
    lat_ptr: np.ndarray,
    lat_nb: np.ndarray,
    rho_long: np.ndarray,
    target_share: np.ndarray,
    edge_rho: np.ndarray,
    lane_edge: np.ndarray,
    vmax: np.ndarray,
    rho_jam: float,
    dt: float,
    k_lc: float,
    rho_next: np.ndarray,
    speed_out: np.ndarray,
    flow_next: np.ndarray,
) -> None:
    for i in range(start, end):
        ei = int(lane_edge[i])
        er = float(edge_rho[ei])
        phi_i = float(rho_long[i]) - float(target_share[i]) * er
        s = int(lat_ptr[i])
        e = int(lat_ptr[i + 1])
        lat = 0.0
        for k in range(s, e):
            j = int(lat_nb[k])
            phi_j = float(rho_long[j]) - float(target_share[j]) * er
            lat += (phi_j - phi_i)

        r = float(rho_long[i]) + dt * k_lc * lat
        if r < 0.0:
            r = 0.0
        if r > rho_jam:
            r = rho_jam
        v = max(float(vmax[i]) * (1.0 - r / rho_jam), 0.0)
        rho_next[i] = r
        speed_out[i] = v
        flow_next[i] = r * v


def run_sim(args) -> None:
    net = load_lane_net(Path(args.net_file))
    n = net.n_lanes

    workers = args.num_workers if args.num_workers > 0 else 8
    workers = max(1, workers)
    chunk = math.ceil(n / workers)
    ranges: list[tuple[int, int]] = []
    for w in range(workers):
        s = w * chunk
        e = min(n, s + chunk)
        if s < e:
            ranges.append((s, e))
    log(f"CPU MT(lane) 설정: lanes={n}, edges={net.n_edges}, workers={workers}, chunks={len(ranges)}")

    rho_jam = float(args.jam_density_per_lane)

    rng = np.random.default_rng(args.seed)
    rho = (args.init_density * rng.uniform(0.7, 1.3, size=n)).astype(np.float32)
    rho = np.clip(rho, 0.0, rho_jam * 0.95)

    rho_long = np.empty_like(rho)
    rho_next = np.empty_like(rho)
    speed = np.zeros_like(rho)
    flow = np.zeros_like(rho)
    flow_next = np.zeros_like(rho)

    source_demand, target_share, veh_n = build_demand_and_target(
        net, Path(args.net_file), Path(args.route_file) if args.route_file else None,
        args.sim_duration, args.source_demand,
    )
    log(f"수요/목표분배 준비 완료: vehicles={veh_n}, lane-change-rate={args.lane_change_rate}")

    lane_edge = net.lane_edge
    edge_rho = np.zeros(net.n_edges, dtype=np.float32)

    def run_step() -> None:
        # phase 1: 종방향
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(
                    longitudinal_chunk, s, e, net.in_ptr, net.in_lanes, net.in_w,
                    rho, flow, net.vmax_mps, rho_jam, net.length_m, args.dt,
                    source_demand, rho_long,
                )
                for s, e in ranges
            ]
            for f in futs:
                f.result()
        # edge별 밀도 합(횡방향 목표 산정용)
        edge_rho[:] = 0.0
        np.add.at(edge_rho, lane_edge, rho_long)
        # phase 2: 횡방향(차선변경) + 최종 상태
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(
                    lateral_chunk, s, e, net.lat_ptr, net.lat_neighbors, rho_long,
                    target_share, edge_rho, lane_edge, net.vmax_mps, rho_jam,
                    args.dt, args.lane_change_rate, rho_next, speed, flow_next,
                )
                for s, e in ranges
            ]
            for f in futs:
                f.result()

    # 초기 1스텝(flow/speed 정렬)
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
                f"step={step+1}/{args.steps} mean_speed={float(np.mean(speed)):.3f} m/s "
                f"mean_density={float(np.mean(rho)):.6f} veh/m"
            )

    elapsed = time.perf_counter() - t0
    log(f"CPU MT(lane) 시뮬레이션 완료: {elapsed:.3f}s ({args.steps} steps)")

    write_outputs(args, net, rho, speed, flow)


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

    # 혼잡 상위 lane
    worst = np.argsort(speed_ratio)[: args.topk]
    log("혼잡 상위 lane")
    for idx in worst:
        log(
            f"  lane={net.lane_ids[idx]} speed={speed[idx]:.3f}/{net.vmax_mps[idx]:.3f} "
            f"ratio={speed_ratio[idx]:.3f} density={rho[idx]:.5f}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Lane-per-thread CPU multithread traffic simulator")
    p.add_argument("--net-file", default="./map_import/gangnam4_generated.net.xml")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--jam-density-per-lane", type=float, default=0.18, help="veh/m/lane")
    p.add_argument("--init-density", type=float, default=0.03, help="veh/m")
    p.add_argument("--source-demand", type=float, default=0.02, help="veh/s")
    p.add_argument("--route-file", default="./map_import/gangnam4_generated.gpu_compatible.rou.xml")
    p.add_argument("--sim-duration", type=float, default=86400.0)
    p.add_argument("--lane-change-rate", type=float, default=0.5, help="횡방향 완화 계수 k_lc")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=200)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--output-csv", default="./gangnam4_cuda/results/lane_state_cpu_mt.csv")
    p.add_argument("--edge-output-csv", default="./gangnam4_cuda/results/lane_state_cpu_mt.edge.csv")
    args = p.parse_args()
    run_sim(args)


if __name__ == "__main__":
    main()
