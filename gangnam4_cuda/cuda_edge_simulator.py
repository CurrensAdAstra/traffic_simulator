#!/usr/bin/env python3
"""
CUDA 기반 edge-병렬(도로 1개 = GPU 스레드 1개) 교통 시뮬레이터.

- SUMO와 분리된 독립 엔진
- 강남4구 net.xml(edge/connection) 입력
- edge별 밀도/속도/유량을 CUDA 커널로 시간 전개
"""

from __future__ import annotations

import argparse
import csv
import os
import time
import xml.etree.ElementTree as ET
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


def build_kernel(cp):
    code = r'''
    extern "C" __global__
    void step_kernel(
        const int n,
        const int* in_ptr,
        const int* in_edges,
        const float* in_w,
        const float* rho,
        const float* flow,
        const float* vmax,
        const float* rho_jam,
        const float* length,
        const float dt,
        const float* source_demand_arr,
        float* rho_next,
        float* speed,
        float* flow_next
    ) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= n) return;

        int s = in_ptr[i];
        int e = in_ptr[i + 1];

        float inflow = source_demand_arr[i];
        if (e > s) {
            inflow = 0.0f;
            for (int k = s; k < e; ++k) {
                inflow += flow[in_edges[k]] * in_w[k];
            }
        }

        float rj = fmaxf(rho_jam[i], 1e-6f);
        float v = vmax[i] * (1.0f - rho[i] / rj);
        if (v < 0.0f) v = 0.0f;
        float outflow = rho[i] * v;

        float r = rho[i] + dt * (inflow - outflow) / fmaxf(length[i], 1.0f);
        if (r < 0.0f) r = 0.0f;
        if (r > rj) r = rj;

        float v2 = vmax[i] * (1.0f - r / rj);
        if (v2 < 0.0f) v2 = 0.0f;
        float q2 = r * v2;

        rho_next[i] = r;
        speed[i] = v2;
        flow_next[i] = q2;
    }
    '''
    return cp.RawKernel(code, "step_kernel")


def run_sim(args) -> None:
    log("진단 후보(6개)")
    for i, c in enumerate(
        [
            "GPU 미탐지/드라이버 문제",
            "cupy 미설치",
            "edge 수 대비 thread/block 설정 비효율",
            "입력 net.xml에서 연결(connection) 누락",
            "모델 파라미터(dt, jam density) 부적절",
            "출력 파일 충돌(동일 파일 동시 기록)",
        ],
        1,
    ):
        log(f"  후보{i}: {c}")
    log("유력 원인 2개 가정")
    log("  가정A: 입력 net의 connection 품질이 낮아 유동이 비현실적")
    log("  가정B: dt/jam-density가 너무 공격적이라 수치 불안정")

    try:
        import cupy as cp  # type: ignore
    except Exception as e:
        fail(f"cupy import 실패: {e}")

    net = load_net(Path(args.net_file))

    n = len(net.edge_ids)
    threads = int(args.threads_per_block)
    blocks = (n + threads - 1) // threads
    log(f"CUDA 설정: edges={n}, threads/block={threads}, blocks={blocks}")

    rng = np.random.default_rng(args.seed)
    rho0 = args.init_density * rng.uniform(0.7, 1.3, size=n).astype(np.float32)

    rho_jam = (args.jam_density_per_lane * net.lanes).astype(np.float32)
    rho0 = np.clip(rho0, 0.0, rho_jam * 0.95)

    rho = cp.asarray(rho0)
    rho_next = cp.empty_like(rho)
    vmax = cp.asarray(net.vmax_mps)
    length = cp.asarray(net.length_m)
    rj = cp.asarray(rho_jam)
    speed = cp.zeros_like(rho)
    flow = cp.zeros_like(rho)
    flow_next = cp.zeros_like(rho)

    in_ptr = cp.asarray(net.in_ptr)
    in_edges = cp.asarray(net.in_edges)
    in_w = cp.asarray(net.in_weights)

    source_demand_np = np.full(n, float(args.source_demand), dtype=np.float32)
    if args.route_file:
        rf = Path(args.route_file)
        if rf.exists():
            route_demand, veh_n = build_source_demand_by_edge(rf, net.edge_to_idx, n, args.sim_duration)
            source_demand_np = route_demand
            log(f"vehicle 조건 반영: route_file={rf}, vehicles={veh_n}, sim_duration={args.sim_duration}")
        else:
            log(f"route_file 없음, fallback source_demand 사용: {rf}")
    source_demand_arr = cp.asarray(source_demand_np)

    kernel = build_kernel(cp)

    # 초기 1스텝으로 flow/speed 정렬
    kernel(
        (blocks,),
        (threads,),
        (
            np.int32(n),
            in_ptr,
            in_edges,
            in_w,
            rho,
            flow,
            vmax,
            rj,
            length,
            np.float32(args.dt),
            source_demand_arr,
            rho_next,
            speed,
            flow_next,
        ),
    )
    rho, rho_next = rho_next, rho
    flow, flow_next = flow_next, flow

    t0 = time.perf_counter()
    for step in range(args.steps):
        kernel(
            (blocks,),
            (threads,),
            (
                np.int32(n),
                in_ptr,
                in_edges,
                in_w,
                rho,
                flow,
                vmax,
                rj,
                length,
                np.float32(args.dt),
                source_demand_arr,
                rho_next,
                speed,
                flow_next,
            ),
        )
        rho, rho_next = rho_next, rho
        flow, flow_next = flow_next, flow

        if (step + 1) % args.log_interval == 0:
            mean_speed = float(cp.mean(speed).get())
            mean_density = float(cp.mean(rho).get())
            log(f"step={step+1}/{args.steps} mean_speed={mean_speed:.3f} m/s mean_density={mean_density:.6f} veh/m")

    cp.cuda.runtime.deviceSynchronize()
    elapsed = time.perf_counter() - t0
    log(f"CUDA 시뮬레이션 완료: {elapsed:.3f}s ({args.steps} steps)")

    speed_np = cp.asnumpy(speed)
    rho_np = cp.asnumpy(rho)
    flow_np = cp.asnumpy(flow)

    speed_ratio = speed_np / np.maximum(net.vmax_mps, 1e-6)
    travel_time = net.length_m / np.maximum(speed_np, 0.1)

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
                    float(rho_np[i]),
                    float(speed_np[i]),
                    float(flow_np[i]),
                    float(speed_ratio[i]),
                    float(travel_time[i]),
                ]
            )

    worst = np.argsort(speed_ratio)[: args.topk]
    log("혼잡 상위 edge")
    for idx in worst:
        log(
            f"  edge={net.edge_ids[idx]} speed={speed_np[idx]:.3f}/{net.vmax_mps[idx]:.3f} "
            f"ratio={speed_ratio[idx]:.3f} density={rho_np[idx]:.5f}"
        )

    log(f"결과 저장: {out}")
    log("[진단검증] 가정A/B 검증용 로그(평균속도/혼잡상위 edge) 출력 완료")


def main() -> None:
    p = argparse.ArgumentParser(description="Edge-per-thread CUDA traffic simulator")
    p.add_argument("--net-file", default="./map_import/gangnam4_fallback.net.xml")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--jam-density-per-lane", type=float, default=0.18, help="veh/m/lane")
    p.add_argument("--init-density", type=float, default=0.03, help="veh/m")
    p.add_argument("--source-demand", type=float, default=0.02, help="veh/s (입력 없는 edge 기본 inflow)")
    p.add_argument("--route-file", default="./map_import/vehicles_100k.sorted2.rou.xml", help="SUMO route 파일(차량 조건 반영)")
    p.add_argument("--sim-duration", type=float, default=86400.0, help="route 기반 수요 환산 시간(초)")
    p.add_argument("--threads-per-block", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=200)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--output-csv", default="./gangnam4_cuda/results/edge_state_cuda.csv")
    args = p.parse_args()
    run_sim(args)


if __name__ == "__main__":
    main()
