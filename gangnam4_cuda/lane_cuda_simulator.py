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


def build_ctm_elementwise(cp):
    """CTM 스텝의 elementwise 블록 5개를 fused ElementwiseKernel로 컴파일.

    각 호출이 단일 CUDA 커널로 합쳐져, 기존 cupy `where/maximum/minimum/multiply/clip` 등
    수십 개 launch가 5개로 줄어든다(스텝당 ~30 → ~9 kernel launch).
    """
    fused_SR = cp.ElementwiseKernel(
        in_params='float32 rho, float32 vmax, float32 rho_jam',
        out_params='float32 S, float32 R',
        operation='''
            float rho_c = rho_jam * 0.5f;
            float q_max_v = vmax * rho_jam * 0.25f;
            float q = rho * vmax * (1.0f - rho / rho_jam);
            S = (rho <= rho_c) ? q : q_max_v;
            R = (rho >= rho_c) ? q : q_max_v;
        ''',
        name='ctm_fused_SR',
    )

    fused_scales = cp.ElementwiseKernel(
        in_params='float32 tot_major, float32 tot_minor, float32 R',
        out_params='float32 scale_major, float32 scale_minor, float32 served_major',
        operation='''
            scale_major = fminf(1.0f, tot_major > 0.0f ? R / fmaxf(tot_major, 1e-12f) : 1.0f);
            served_major = tot_major * scale_major;
            float residual = fmaxf(R - served_major, 0.0f);
            scale_minor = fminf(1.0f, tot_minor > 0.0f ? residual / fmaxf(tot_minor, 1e-12f) : 1.0f);
        ''',
        name='ctm_fused_scales',
    )

    fused_rho_long = cp.ElementwiseKernel(
        in_params='float32 rho, float32 served_major, float32 scale_minor, float32 tot_minor, '
                  'float32 outflow, float32 source_demand, int8 no_incoming, '
                  'float32 length_eff, float32 dt, float32 rho_jam',
        out_params='float32 rho_long',
        operation='''
            float inflow = served_major + scale_minor * tot_minor;
            if (no_incoming) inflow += source_demand;
            float r = rho + dt * (inflow - outflow) / length_eff;
            if (r < 0.0f) r = 0.0f;
            if (r > rho_jam) r = rho_jam;
            rho_long = r;
        ''',
        name='ctm_fused_rho_long',
    )

    fused_phi = cp.ElementwiseKernel(
        in_params='float32 rho_long, float32 target_share, raw float32 edge_rho, int32 lane_edge',
        out_params='float32 phi',
        operation='phi = rho_long - target_share * edge_rho[lane_edge];',
        name='ctm_fused_phi',
    )

    fused_final = cp.ElementwiseKernel(
        in_params='float32 rho_long, float32 sum_phi_nb, float32 phi, float32 deg, '
                  'float32 vmax, float32 dt, float32 k_lc, float32 rho_jam',
        out_params='float32 rho_next, float32 speed, float32 flow_next',
        operation='''
            float lat = sum_phi_nb - phi * deg;
            float r = rho_long + dt * k_lc * lat;
            if (r < 0.0f) r = 0.0f;
            if (r > rho_jam) r = rho_jam;
            float v = vmax * (1.0f - r / rho_jam);
            if (v < 0.0f) v = 0.0f;
            rho_next = r;
            speed = v;
            flow_next = r * v;
        ''',
        name='ctm_fused_final',
    )

    return fused_SR, fused_scales, fused_rho_long, fused_phi, fused_final


def build_ctm_kernels(cp):
    """gather-only CTM 커널 묶음 — 모든 scatter_add(atomicAdd)를 제거.

    각 커널은 1 thread = 1 lane(또는 1 edge)으로, 자기 CSR slice만 읽음.
    """
    # Pass A: 각 수신 lane이 자기 incoming 연결을 모아 tot_major/tot_minor 계산
    k_gather_demand = cp.RawKernel(r'''
    extern "C" __global__
    void k_gather_demand(
        const int n,
        const int* __restrict__ in_ptr,
        const int* __restrict__ in_idx,
        const float* __restrict__ S,
        const float* __restrict__ conn_split,
        const int* __restrict__ conn_src,
        const signed char* __restrict__ pri,
        float* __restrict__ tot_major,
        float* __restrict__ tot_minor
    ) {
        int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i >= n) return;
        int s = in_ptr[i], e = in_ptr[i+1];
        float maj = 0.0f, min_ = 0.0f;
        for (int kk = s; kk < e; ++kk) {
            int ci = in_idx[kk];
            float D = S[conn_src[ci]] * conn_split[ci];
            if (pri[ci]) maj += D; else min_ += D;
        }
        tot_major[i] = maj;
        tot_minor[i] = min_;
    }
    ''', "k_gather_demand")

    # Pass B: 각 송신 lane이 자기 outgoing 연결을 모아 outflow 계산
    #   outflow[i] = S[i] * sum_k conn_split[k] * scale_at_dst(k)
    #   + sink drain if no_outgoing
    k_gather_outflow = cp.RawKernel(r'''
    extern "C" __global__
    void k_gather_outflow(
        const int n,
        const int* __restrict__ out_ptr,
        const int* __restrict__ out_idx,
        const float* __restrict__ S,
        const float* __restrict__ conn_split,
        const int* __restrict__ conn_dst,
        const signed char* __restrict__ pri,
        const float* __restrict__ scale_major,
        const float* __restrict__ scale_minor,
        const signed char* __restrict__ no_outgoing,
        float* __restrict__ outflow
    ) {
        int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i >= n) return;
        int s = out_ptr[i], e = out_ptr[i+1];
        float Si = S[i];
        float acc = 0.0f;
        for (int kk = s; kk < e; ++kk) {
            int ci = out_idx[kk];
            int dst = conn_dst[ci];
            float scl = pri[ci] ? scale_major[dst] : scale_minor[dst];
            acc += conn_split[ci] * scl;
        }
        float out_v = Si * acc;
        if (no_outgoing[i]) out_v += Si;
        outflow[i] = out_v;
    }
    ''', "k_gather_outflow")

    # Pass C: edge별로 자기 lane의 rho_long 합 — edge_lane_ptr CSR
    k_gather_edge_rho = cp.RawKernel(r'''
    extern "C" __global__
    void k_gather_edge_rho(
        const int n_edges,
        const int* __restrict__ edge_lane_ptr,
        const int* __restrict__ edge_lanes,
        const float* __restrict__ rho_long,
        float* __restrict__ edge_rho
    ) {
        int e = blockDim.x*blockIdx.x + threadIdx.x;
        if (e >= n_edges) return;
        int s = edge_lane_ptr[e], en = edge_lane_ptr[e+1];
        float acc = 0.0f;
        for (int kk = s; kk < en; ++kk) acc += rho_long[edge_lanes[kk]];
        edge_rho[e] = acc;
    }
    ''', "k_gather_edge_rho")

    # Pass D: 횡방향 — 각 lane이 자기 lat 이웃의 phi 합 (lat_ptr CSR by source)
    k_gather_lat = cp.RawKernel(r'''
    extern "C" __global__
    void k_gather_lat(
        const int n,
        const int* __restrict__ lat_ptr,
        const int* __restrict__ lat_nb,
        const float* __restrict__ phi,
        float* __restrict__ sum_phi_nb,
        float* __restrict__ deg
    ) {
        int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i >= n) return;
        int s = lat_ptr[i], e = lat_ptr[i+1];
        float acc = 0.0f;
        for (int kk = s; kk < e; ++kk) acc += phi[lat_nb[kk]];
        sum_phi_nb[i] = acc;
        deg[i] = (float)(e - s);
    }
    ''', "k_gather_lat")

    return k_gather_demand, k_gather_outflow, k_gather_edge_rho, k_gather_lat


def ctm_step_gpu(cp, kernels, e_kernels, blocks, threads, blocks_e, net_g,
                  rho, vmax, rho_jam, length_eff, dt, source_demand, target_share, k_lc,
                  no_incoming_mask, no_outgoing_mask, conn_split, bufs,
                  rho_next_buf, speed_next_buf, flow_next_buf):
    """GPU CTM 한 스텝 — gather RawKernel + fused ElementwiseKernel.

    스텝당 커널 launch ≈ 9개(이전 ~30+). 모든 출력은 caller가 미리 할당한
    고정 버퍼에 in-place로 기록 → CUDA Graphs 캡처도 가능한 구조.
    """
    n = rho.shape[0]
    k_gather_demand, k_gather_outflow, k_gather_edge_rho, k_gather_lat = kernels
    fused_SR, fused_scales, fused_rho_long, fused_phi, fused_final = e_kernels

    rho_jam_f32 = np.float32(rho_jam)
    dt_f32 = np.float32(dt)
    k_lc_f32 = np.float32(k_lc)

    S = bufs["S"]; R = bufs["R"]
    tot_major = bufs["tot_major"]; tot_minor = bufs["tot_minor"]
    scale_major = bufs["scale_major"]; scale_minor = bufs["scale_minor"]
    served_major = bufs["served_major"]
    outflow = bufs["outflow"]
    rho_long = bufs["rho_long"]
    edge_rho = bufs["edge_rho"]
    phi = bufs["phi"]
    sum_phi_nb = bufs["sum_phi_nb"]; deg = bufs["deg"]

    # 1) S, R (fused)
    fused_SR(rho, vmax, rho_jam_f32, S, R)

    # 2) gather demand → tot_major, tot_minor
    k_gather_demand(
        (blocks,), (threads,),
        (np.int32(n), net_g["in_conn_ptr"], net_g["in_conn_idx"],
         S, conn_split, net_g["conn_src"], net_g["conn_priority"],
         tot_major, tot_minor),
    )

    # 3) scales (fused)
    fused_scales(tot_major, tot_minor, R, scale_major, scale_minor, served_major)

    # 4) gather outflow (with sink drain)
    k_gather_outflow(
        (blocks,), (threads,),
        (np.int32(n), net_g["out_conn_ptr"], net_g["out_conn_idx"],
         S, conn_split, net_g["conn_dst"], net_g["conn_priority"],
         scale_major, scale_minor, no_outgoing_mask, outflow),
    )

    # 5) rho_long = clip(rho + dt*(inflow-outflow)/length_eff)  (fused; inflow inline)
    fused_rho_long(
        rho, served_major, scale_minor, tot_minor, outflow,
        source_demand, no_incoming_mask, length_eff,
        dt_f32, rho_jam_f32, rho_long,
    )

    # 6) edge_rho gather
    k_gather_edge_rho(
        (blocks_e,), (threads,),
        (np.int32(net_g["n_edges"]), net_g["edge_lane_ptr"], net_g["edge_lanes"],
         rho_long, edge_rho),
    )

    # 7) phi (fused, with edge_rho[lane_edge] indexing inside)
    fused_phi(rho_long, target_share, edge_rho, net_g["lane_edge"], phi)

    # 8) lateral gather → sum_phi_nb, deg
    k_gather_lat(
        (blocks,), (threads,),
        (np.int32(n), net_g["lat_ptr"], net_g["lat_neighbors"], phi,
         sum_phi_nb, deg),
    )

    # 9) final: lat → rho_next, speed, flow_next (fused)
    fused_final(
        rho_long, sum_phi_nb, phi, deg, vmax,
        dt_f32, k_lc_f32, rho_jam_f32,
        rho_next_buf, speed_next_buf, flow_next_buf,
    )


def run_sim(args) -> None:
    try:
        import cupy as cp  # type: ignore
        import cupyx  # type: ignore
        from cupyx import scatter_add  # type: ignore
    except Exception as e:
        fail(f"cupy import 실패: {e}")

    net = load_lane_net(Path(args.net_file))
    n = net.n_lanes
    threads = int(args.threads_per_block)
    blocks = (n + threads - 1) // threads
    model = args.model.lower()
    if model not in {"lwr", "ctm"}:
        raise SystemExit(f"unknown --model: {args.model}")

    # --sim-time이 주어지면 steps를 sim_time/dt로 재계산
    steps = args.steps
    if args.sim_time and args.sim_time > 0:
        steps = max(1, int(round(args.sim_time / args.dt)))
        log(f"sim_time={args.sim_time}s 적용: steps={steps} (dt={args.dt})")
    log(f"CUDA(lane,{model}) 설정: lanes={n}, edges={net.n_edges}, "
        f"threads/block={threads}, blocks={blocks}, time_average={bool(args.time_average)}, steps={steps}")

    rho_jam = np.float32(args.jam_density_per_lane)

    source_demand_np, target_share_np, conn_split_cal_np, veh_n = build_demand_and_target(
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

    # CTM 전용 디바이스 자료구조 — gather 커널이 직접 읽는 CSR 포함
    net_g = {
        "n_edges": net.n_edges,
        "lane_edge": lane_edge,
        "lat_ptr": lat_ptr,
        "lat_neighbors": lat_nb,
        "conn_src": cp.asarray(net.conn_src),
        "conn_dst": cp.asarray(net.conn_dst),
        "conn_priority": cp.asarray(net.conn_priority),
        "in_conn_ptr": cp.asarray(net.in_conn_ptr),
        "in_conn_idx": cp.asarray(net.in_conn_idx),
        "out_conn_ptr": cp.asarray(net.out_conn_ptr),
        "out_conn_idx": cp.asarray(net.out_conn_idx),
        "edge_lane_ptr": cp.asarray(net.edge_lane_ptr),
        "edge_lanes": cp.asarray(net.edge_lanes),
    }
    conn_split_cal = cp.asarray(conn_split_cal_np)
    out_count = cp.bincount(net_g["conn_src"], minlength=n)
    # int8 마스크(RawKernel signed char로 받음)
    no_outgoing_mask = (out_count == 0).astype(cp.int8)
    no_incoming_mask = ((in_ptr[1:] - in_ptr[:-1]) == 0).astype(cp.int8)
    # CFL-안정 effective length
    length_eff = cp.maximum(length, np.float32(args.dt) * vmax)
    log(f"진입 lane={int(no_incoming_mask.sum().get())}, 출구 lane={int(no_outgoing_mask.sum().get())}")

    long_kernel, lat_kernel = build_kernels(cp)
    ctm_kernels = build_ctm_kernels(cp)
    ctm_e_kernels = build_ctm_elementwise(cp)
    blocks_e = (net.n_edges + threads - 1) // threads
    # CTM step에서 재사용할 버퍼들 — fused 커널의 in-place 출력 + ping-pong용
    ctm_bufs = {
        "S":           cp.empty(n, dtype=cp.float32),
        "R":           cp.empty(n, dtype=cp.float32),
        "tot_major":   cp.empty(n, dtype=cp.float32),
        "tot_minor":   cp.empty(n, dtype=cp.float32),
        "scale_major": cp.empty(n, dtype=cp.float32),
        "scale_minor": cp.empty(n, dtype=cp.float32),
        "served_major": cp.empty(n, dtype=cp.float32),
        "outflow":     cp.empty(n, dtype=cp.float32),
        "rho_long":    cp.empty(n, dtype=cp.float32),
        "edge_rho":    cp.empty(net.n_edges, dtype=cp.float32),
        "phi":         cp.empty(n, dtype=cp.float32),
        "sum_phi_nb":  cp.empty(n, dtype=cp.float32),
        "deg":         cp.empty(n, dtype=cp.float32),
    }

    def run_step_lwr() -> None:
        long_kernel(
            (blocks,), (threads,),
            (np.int32(n), in_ptr, in_lanes, in_w, rho, flow, vmax, rho_jam,
             length, np.float32(args.dt), source_demand, rho_long),
        )
        edge_rho.fill(0)
        scatter_add(edge_rho, lane_edge, rho_long)
        lat_kernel(
            (blocks,), (threads,),
            (np.int32(n), lat_ptr, lat_nb, rho_long, target_share, edge_rho,
             lane_edge, vmax, rho_jam, np.float32(args.dt),
             np.float32(args.lane_change_rate), rho_next, speed, flow_next),
        )

    # 시간평균 누적기
    rho_acc = cp.zeros(n, dtype=cp.float64) if args.time_average else None
    speed_acc = cp.zeros(n, dtype=cp.float64) if args.time_average else None
    flow_acc = cp.zeros(n, dtype=cp.float64) if args.time_average else None
    t_acc = 0.0

    # CTM용 ping-pong 출력 버퍼 (rho/speed/flow의 alternate)
    rho_alt = cp.empty_like(rho)
    speed_alt = cp.empty_like(speed)
    flow_alt = cp.empty_like(flow)

    def ctm_one_step():
        nonlocal rho, speed, flow, rho_alt, speed_alt, flow_alt
        ctm_step_gpu(
            cp, ctm_kernels, ctm_e_kernels, blocks, threads, blocks_e, net_g,
            rho, vmax, float(rho_jam), length_eff, float(args.dt),
            source_demand, target_share, float(args.lane_change_rate),
            no_incoming_mask, no_outgoing_mask, conn_split_cal, ctm_bufs,
            rho_alt, speed_alt, flow_alt,
        )
        rho, rho_alt = rho_alt, rho
        speed, speed_alt = speed_alt, speed
        flow, flow_alt = flow_alt, flow

    # 초기 1스텝(flow/speed 정렬)
    if model == "lwr":
        run_step_lwr()
        rho, rho_next = rho_next, rho
        flow, flow_next = flow_next, flow
    else:
        ctm_one_step()

    t0 = time.perf_counter()
    for step in range(steps):
        if model == "lwr":
            run_step_lwr()
            rho, rho_next = rho_next, rho
            flow, flow_next = flow_next, flow
        else:
            ctm_one_step()

        if args.time_average:
            rho_acc += rho.astype(cp.float64) * args.dt
            speed_acc += speed.astype(cp.float64) * args.dt
            flow_acc += flow.astype(cp.float64) * args.dt
            t_acc += args.dt

        if (step + 1) % args.log_interval == 0:
            log(
                f"step={step+1}/{steps} mean_speed={float(cp.mean(speed).get()):.3f} m/s "
                f"mean_density={float(cp.mean(rho).get()):.6f} veh/m"
            )

    cp.cuda.runtime.deviceSynchronize()
    elapsed = time.perf_counter() - t0
    log(f"CUDA(lane,{model}) 시뮬레이션 완료: {elapsed:.3f}s ({steps} steps)")

    if args.time_average and t_acc > 0:
        rho = (rho_acc / t_acc).astype(cp.float32)
        speed = (speed_acc / t_acc).astype(cp.float32)
        flow = (flow_acc / t_acc).astype(cp.float32)
        log(f"시간평균 적용: T={t_acc:.1f}s 평균값으로 출력")

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
    p.add_argument("--model", default="lwr", choices=["lwr", "ctm"],
                   help="갱신 모델 — lwr(기본, 단순) 또는 ctm(spillback+priority+route-split)")
    p.add_argument("--time-average", action="store_true",
                   help="스텝별 (rho,speed,flow)을 시간 평균하여 출력")
    p.add_argument("--sim-time", type=float, default=0.0,
                   help="모델 시뮬레이션 시간(초). >0이면 steps를 sim_time/dt로 재계산")
    args = p.parse_args()
    run_sim(args)


if __name__ == "__main__":
    main()
