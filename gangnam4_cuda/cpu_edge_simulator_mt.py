#!/usr/bin/env python3
"""
CPU 멀티스레딩(edge-병렬) 교통 시뮬레이터.

- SUMO와 분리된 독립 엔진
- CUDA 버전(`cuda_edge_simulator.py`)과 동일한 입력/출력 철학
- 각 edge를 청크로 나눠 스레드 풀에서 병렬 갱신
"""

from __future__ import annotations

import argparse
import csv
import math
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def log(msg: str) -> None:
    print(f"[LOG] {msg}")


def fail(msg: str) -> None:
    print(f"[ERROR] {msg}")
    raise SystemExit(1)


@dataclass
class NetData:
    edge_ids: list[str]
    edge_to_idx: dict[str, int]
    length_m: np.ndarray
    lanes: np.ndarray
    vmax_mps: np.ndarray
    in_ptr: np.ndarray
    in_edges: np.ndarray
    in_weights: np.ndarray


def load_net(net_file: Path) -> NetData:
    if not net_file.exists():
        fail(f"net 파일 없음: {net_file}")

    root = ET.parse(net_file).getroot()
    edge_ids: list[str] = []
    length: list[float] = []
    lanes: list[float] = []
    vmax: list[float] = []

    for e in root.findall("edge"):
        eid = e.get("id", "")
        if not eid or eid.startswith(":"):
            continue
        if e.get("function", "") in {"internal", "crossing", "walkingarea"}:
            continue

        lane_nodes = e.findall("lane")
        if not lane_nodes:
            continue

        l0 = float(lane_nodes[0].get("length", "1"))
        v0 = float(lane_nodes[0].get("speed", "13.9"))
        edge_ids.append(eid)
        length.append(max(l0, 1.0))
        lanes.append(float(len(lane_nodes)))
        vmax.append(max(v0, 0.1))

    n = len(edge_ids)
    if n == 0:
        fail("유효 edge가 0개")

    edge_to_idx = {eid: i for i, eid in enumerate(edge_ids)}

    outgoing_count = np.zeros(n, dtype=np.int32)
    raw_conns: list[tuple[int, int]] = []
    for c in root.findall("connection"):
        f = c.get("from", "")
        t = c.get("to", "")
        if f in edge_to_idx and t in edge_to_idx:
            fi, ti = edge_to_idx[f], edge_to_idx[t]
            raw_conns.append((fi, ti))
            outgoing_count[fi] += 1

    incoming_lists: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for fi, ti in raw_conns:
        w = 1.0 / max(outgoing_count[fi], 1)
        incoming_lists[ti].append((fi, float(w)))

    in_ptr = [0]
    in_edges: list[int] = []
    in_weights: list[float] = []
    for arr in incoming_lists:
        for fi, w in arr:
            in_edges.append(fi)
            in_weights.append(w)
        in_ptr.append(len(in_edges))

    log(f"net 로드 완료: edges={n}, connections={len(raw_conns)}")
    return NetData(
        edge_ids=edge_ids,
        edge_to_idx=edge_to_idx,
        length_m=np.asarray(length, dtype=np.float32),
        lanes=np.asarray(lanes, dtype=np.float32),
        vmax_mps=np.asarray(vmax, dtype=np.float32),
        in_ptr=np.asarray(in_ptr, dtype=np.int32),
        in_edges=np.asarray(in_edges, dtype=np.int32),
        in_weights=np.asarray(in_weights, dtype=np.float32),
    )


def build_source_demand_by_edge(route_file: Path, edge_to_idx: dict[str, int], n: int, sim_duration: float) -> tuple[np.ndarray, int]:
    demand = np.zeros(n, dtype=np.float32)
    vehicle_count = 0

    if not route_file.exists():
        return demand, vehicle_count

    for event, elem in ET.iterparse(route_file, events=("end",)):
        if elem.tag == "vehicle":
            vehicle_count += 1
            route_node = elem.find("route")
            if route_node is not None:
                edges_str = route_node.get("edges", "").strip()
                if edges_str:
                    first_edge = edges_str.split()[0]
                    idx = edge_to_idx.get(first_edge)
                    if idx is not None:
                        demand[idx] += 1.0
            elem.clear()

    if sim_duration <= 0:
        sim_duration = 1.0
    demand /= float(sim_duration)
    return demand, vehicle_count


def step_chunk(
    start: int,
    end: int,
    in_ptr: np.ndarray,
    in_edges: np.ndarray,
    in_weights: np.ndarray,
    rho: np.ndarray,
    flow: np.ndarray,
    vmax: np.ndarray,
    rho_jam: np.ndarray,
    length: np.ndarray,
    dt: float,
    source_demand_arr: np.ndarray,
    rho_next: np.ndarray,
    speed_out: np.ndarray,
    flow_next: np.ndarray,
) -> None:
    for i in range(start, end):
        s = int(in_ptr[i])
        e = int(in_ptr[i + 1])
        if e > s:
            inflow = float(np.dot(flow[in_edges[s:e]], in_weights[s:e]))
        else:
            inflow = float(source_demand_arr[i])

        rj = max(float(rho_jam[i]), 1e-6)
        v = max(float(vmax[i]) * (1.0 - float(rho[i]) / rj), 0.0)
        outflow = float(rho[i]) * v

        r = float(rho[i]) + dt * (inflow - outflow) / max(float(length[i]), 1.0)
        if r < 0.0:
            r = 0.0
        if r > rj:
            r = rj

        v2 = max(float(vmax[i]) * (1.0 - r / rj), 0.0)
        q2 = r * v2

        rho_next[i] = r
        speed_out[i] = v2
        flow_next[i] = q2


def run_sim(args) -> None:
    log("진단 후보(6개)")
    for i, c in enumerate(
        [
            "스레드 수 과다로 컨텍스트 스위칭 증가",
            "입력 net.xml의 connection 품질 문제",
            "dt/jam-density 설정 부적절",
            "파이썬 루프 오버헤드로 GPU 대비 저속",
            "출력 파일 충돌(동일 파일 동시 기록)",
            "NUMA/BLAS 스레드 설정 충돌",
        ],
        1,
    ):
        log(f"  후보{i}: {c}")
    log("유력 원인 2개 가정")
    log("  가정A: num_workers와 edge 수 비율이 비효율")
    log("  가정B: dt/jam-density가 현재 네트워크에 비적합")

    net = load_net(Path(args.net_file))
    n = len(net.edge_ids)

    workers = args.num_workers if args.num_workers > 0 else max(1, (Path('/proc/cpuinfo').read_text().count('processor\t:') if Path('/proc/cpuinfo').exists() else 8))
    workers = max(1, workers)

    chunk = math.ceil(n / workers)
    ranges: list[tuple[int, int]] = []
    for w in range(workers):
        s = w * chunk
        e = min(n, s + chunk)
        if s < e:
            ranges.append((s, e))

    log(f"CPU MT 설정: edges={n}, workers={workers}, chunks={len(ranges)}")

    rng = np.random.default_rng(args.seed)
    rho_jam = (args.jam_density_per_lane * net.lanes).astype(np.float32)
    rho = (args.init_density * rng.uniform(0.7, 1.3, size=n)).astype(np.float32)
    rho = np.clip(rho, 0.0, rho_jam * 0.95)

    rho_next = np.empty_like(rho)
    speed = np.zeros_like(rho)
    flow = np.zeros_like(rho)
    flow_next = np.zeros_like(rho)

    source_demand_arr = np.full(n, float(args.source_demand), dtype=np.float32)
    if args.route_file:
        rf = Path(args.route_file)
        if rf.exists():
            route_demand, veh_n = build_source_demand_by_edge(rf, net.edge_to_idx, n, args.sim_duration)
            source_demand_arr = route_demand
            log(f"vehicle 조건 반영: route_file={rf}, vehicles={veh_n}, sim_duration={args.sim_duration}")
        else:
            log(f"route_file 없음, fallback source_demand 사용: {rf}")

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for step in range(args.steps):
            futures = [
                ex.submit(
                    step_chunk,
                    s,
                    e,
                    net.in_ptr,
                    net.in_edges,
                    net.in_weights,
                    rho,
                    flow,
                    net.vmax_mps,
                    rho_jam,
                    net.length_m,
                    args.dt,
                    source_demand_arr,
                    rho_next,
                    speed,
                    flow_next,
                )
                for s, e in ranges
            ]
            for f in futures:
                f.result()

            rho, rho_next = rho_next, rho
            flow, flow_next = flow_next, flow

            if (step + 1) % args.log_interval == 0:
                log(
                    f"step={step+1}/{args.steps} mean_speed={float(np.mean(speed)):.3f} m/s "
                    f"mean_density={float(np.mean(rho)):.6f} veh/m"
                )

    elapsed = time.perf_counter() - t0
    log(f"CPU 멀티스레딩 시뮬레이션 완료: {elapsed:.3f}s ({args.steps} steps)")

    speed_ratio = speed / np.maximum(net.vmax_mps, 1e-6)
    travel_time = net.length_m / np.maximum(speed, 0.1)

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["edge_id", "length_m", "lanes", "vmax_mps", "density_veh_per_m", "speed_mps", "flow_veh_per_s", "speed_ratio", "travel_time_s"])
        for i, eid in enumerate(net.edge_ids):
            w.writerow(
                [
                    eid,
                    float(net.length_m[i]),
                    float(net.lanes[i]),
                    float(net.vmax_mps[i]),
                    float(rho[i]),
                    float(speed[i]),
                    float(flow[i]),
                    float(speed_ratio[i]),
                    float(travel_time[i]),
                ]
            )

    worst = np.argsort(speed_ratio)[: args.topk]
    log("혼잡 상위 edge")
    for idx in worst:
        log(
            f"  edge={net.edge_ids[idx]} speed={speed[idx]:.3f}/{net.vmax_mps[idx]:.3f} "
            f"ratio={speed_ratio[idx]:.3f} density={rho[idx]:.5f}"
        )

    log(f"결과 저장: {out}")
    log("[진단검증] 가정A/B 검증용 로그(평균속도/혼잡상위 edge) 출력 완료")


def main() -> None:
    p = argparse.ArgumentParser(description="Edge-per-thread CPU multithread traffic simulator")
    p.add_argument("--net-file", default="./map_import/gangnam4_fallback.net.xml")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--jam-density-per-lane", type=float, default=0.18, help="veh/m/lane")
    p.add_argument("--init-density", type=float, default=0.03, help="veh/m")
    p.add_argument("--source-demand", type=float, default=0.02, help="veh/s")
    p.add_argument("--route-file", default="./map_import/vehicles_100k.sorted2.rou.xml", help="SUMO route 파일(차량 조건 반영)")
    p.add_argument("--sim-duration", type=float, default=86400.0, help="route 기반 수요 환산 시간(초)")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=200)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--output-csv", default="./gangnam4_cuda/results/edge_state_cpu_mt.csv")
    args = p.parse_args()
    run_sim(args)


if __name__ == "__main__":
    main()
