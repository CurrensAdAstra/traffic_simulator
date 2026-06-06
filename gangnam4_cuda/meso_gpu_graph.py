#!/usr/bin/env python3
"""
Branch-free + CUDA Graphs 메소스코픽 GPU 엔진 (meso_gpu.py 재설계).

프로파일 결과: meso-GPU의 10× 천장은 lexsort가 아니라 **수십 개의 작은 커널 launch +
host-device 동기화**(flatnonzero/.any()/.size)에서 옴. 이를 제거:
  - 모든 스텝 연산을 전 차량(V) 고정크기 커널로 처리(in-kernel 분기, atomic ticket)
  - flatnonzero/.any()/.size/scalar-read 전부 제거 → 데이터 의존 분기 없음
  - 시각 t는 device 스칼라(tick 커널이 갱신) → 인자 baking 문제 회피
  - 스텝 전체를 단일 CUDA Graph로 캡처해 replay (launch overhead 제거)

이 패턴이 CTM에서 1.36s→0.09s(15×)를 만든 그 방법. capacity/space는 스냅샷 + atomic
ticket으로 정렬 없이 처리(메소 근사; 그룹 내 순서는 비결정적).
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from meso_common import load_meso_net, log, fail

KERNELS = r'''
extern "C" {

__global__ void k_tick(const float dt, int* step_dev, double* t_dev){
    // 1 thread
    int s = step_dev[0];
    t_dev[0] = (double)s * (double)dt;
    step_dev[0] = s + 1;
}

__global__ void k_edge_speed(const int E, const int* edge_count, const double* length,
        const double* lanes, const double* vmax, const double rho_jam,
        const double min_speed, double* espeed){
    int e = blockDim.x*blockIdx.x + threadIdx.x; if(e>=E) return;
    double dens = (double)edge_count[e] / fmax(length[e]*lanes[e], 1.0);
    double v = vmax[e]*(1.0 - dens/rho_jam);
    espeed[e] = fmax(v, min_speed);
}

__global__ void k_advance(const int V, const double* t_dev, const double dt,
        signed char* state, const int* cur_edge, double* pos_m, const double* hold_until,
        const double* espeed, const double* length){
    int i = blockDim.x*blockIdx.x + threadIdx.x; if(i>=V) return;
    if(state[i]!=1) return;                 // RUN
    double t = t_dev[0];
    if(hold_until[i] > t) return;
    int e = cur_edge[i];
    pos_m[i] += espeed[e]*dt;
    if(pos_m[i] >= length[e]) state[i]=2;   // QUEUE
}

__global__ void k_credit(const int E, const double* sat_cap, const double dt, double* out_credit){
    int e = blockDim.x*blockIdx.x + threadIdx.x; if(e>=E) return;
    out_credit[e] += sat_cap[e]*dt;
}

__global__ void k_send_ticket(const int V, const signed char* state, const int* cur_edge,
        int* send_counter, int* send_rank){
    int i = blockDim.x*blockIdx.x + threadIdx.x; if(i>=V) return;
    if(state[i]==2){ send_rank[i] = atomicAdd(&send_counter[cur_edge[i]], 1); }
    else send_rank[i] = 2000000000;
}

__global__ void k_recv(const int V, const signed char* state, const int* cur_edge,
        const int* cur_pos, const int* veh_off, const int* veh_len, const int* redges,
        const double* out_credit, const int* send_rank,
        int* recv_counter, int* next_edge, int* recv_rank){
    int i = blockDim.x*blockIdx.x + threadIdx.x; if(i>=V) return;
    next_edge[i] = -2; recv_rank[i] = 2000000000;
    if(state[i]!=2) return;
    int e = cur_edge[i];
    int sc = (int)floor(out_credit[e]);
    if(send_rank[i] < sc){
        int np1 = cur_pos[i]+1;
        if(np1 >= veh_len[i]){ next_edge[i] = -1; }   // exit
        else { int ne = redges[veh_off[i]+np1]; next_edge[i]=ne;
               recv_rank[i] = atomicAdd(&recv_counter[ne], 1); }
    }
}

__global__ void k_apply_disch(const int V, const double* t_dev, const double jct_delay,
        signed char* state, int* cur_edge, int* cur_pos, const int* next_edge,
        const int* recv_rank, const double* jam_storage, const int* edge_count_snap,
        int* edge_count, double* out_credit, double* edge_exits, double* arrival_time,
        double* route_dist, const double* length, double* pos_m, double* enter_time,
        double* hold_until){
    int i = blockDim.x*blockIdx.x + threadIdx.x; if(i>=V) return;
    int ne = next_edge[i];
    if(ne == -2) return;                       // not sending
    bool admitted;
    if(ne == -1) admitted = true;              // exit always
    else { int space = (int)floor(jam_storage[ne]) - edge_count_snap[ne];
           admitted = recv_rank[i] < space; }
    if(!admitted) return;
    double t = t_dev[0];
    int e = cur_edge[i];
    atomicAdd(&edge_count[e], -1);
    atomicAdd(&out_credit[e], -1.0);
    atomicAdd(&edge_exits[e], 1.0);
    if(ne == -1){ state[i]=3; arrival_time[i]=t; cur_edge[i]=-1; }   // DONE
    else {
        route_dist[i] += length[e];
        cur_pos[i] += 1; cur_edge[i]=ne; enter_time[i]=t; pos_m[i]=0.0;
        hold_until[i]=t+jct_delay;
        atomicAdd(&edge_count[ne], 1); state[i]=1;                  // RUN
    }
}

__global__ void k_dep_ticket(const int V, const double* t_dev, const signed char* state,
        const double* veh_depart, const int* veh_off, const int* redges,
        int* dep_counter, int* first_edge, int* dep_rank){
    int i = blockDim.x*blockIdx.x + threadIdx.x; if(i>=V) return;
    first_edge[i] = -2; dep_rank[i] = 2000000000;
    if(state[i]!=0) return;                    // PRE
    if(veh_depart[i] > t_dev[0]) return;
    int fe = redges[veh_off[i]];
    first_edge[i] = fe;
    dep_rank[i] = atomicAdd(&dep_counter[fe], 1);
}

__global__ void k_dep_apply(const int V, const double* t_dev, signed char* state,
        const int* first_edge, const int* dep_rank, const double* jam_storage,
        const int* edge_count_snap2, int* edge_count, int* cur_pos, int* cur_edge,
        double* enter_time, double* start_time, double* pos_m){
    int i = blockDim.x*blockIdx.x + threadIdx.x; if(i>=V) return;
    int fe = first_edge[i];
    if(fe < 0) return;
    int space = (int)floor(jam_storage[fe]) - edge_count_snap2[fe];
    if(dep_rank[i] < space){
        double t = t_dev[0];
        state[i]=1; cur_pos[i]=0; cur_edge[i]=fe; enter_time[i]=t; start_time[i]=t;
        pos_m[i]=0.0; atomicAdd(&edge_count[fe], 1);
    }
}

__global__ void k_accum(const int E, const int* edge_count, const double dt, double* acc_count){
    int e = blockDim.x*blockIdx.x + threadIdx.x; if(e>=E) return;
    acc_count[e] += (double)edge_count[e]*dt;
}
}
'''


def run_sim(args) -> None:
    try:
        import cupy as cp  # type: ignore
    except Exception as e:
        fail(f"cupy import 실패: {e}")

    net = load_meso_net(Path(args.net_file), Path(args.route_file),
                        jam_density_per_lane=args.jam_density_per_lane,
                        sat_flow_per_lane=args.sat_flow_per_lane,
                        vmax_scale=args.vmax_scale, max_vehicles=args.max_vehicles)
    V, E = net.n_veh, net.n_edges
    if V == 0:
        fail("차량 0대")
    dt = float(args.dt); n_steps = int(round(args.sim_time / dt))
    rho_jam = float(args.jam_density_per_lane)
    jct_delay = float(args.junction_delay)

    mod = cp.RawModule(code=KERNELS, options=('--use_fast_math',))
    K = {name: mod.get_function(name) for name in
         ["k_tick","k_edge_speed","k_advance","k_credit","k_send_ticket","k_recv",
          "k_apply_disch","k_dep_ticket","k_dep_apply","k_accum"]}

    # 디바이스 배열
    length = cp.asarray(net.length_m, dtype=cp.float64)
    lanes = cp.asarray(net.lanes, dtype=cp.float64)
    vmax = cp.asarray(net.vmax_mps, dtype=cp.float64)
    jam_storage = cp.asarray(net.jam_storage, dtype=cp.float64)
    sat_cap = cp.asarray(net.sat_cap_per_s, dtype=cp.float64)
    veh_off = cp.asarray(net.veh_route_off, dtype=cp.int32)
    veh_len = cp.asarray(net.veh_route_len, dtype=cp.int32)
    redges = cp.asarray(net.route_edges, dtype=cp.int32)
    veh_depart = cp.asarray(net.veh_depart, dtype=cp.float64)

    state = cp.zeros(V, dtype=cp.int8)
    cur_pos = cp.full(V, -1, dtype=cp.int32)
    cur_edge = cp.full(V, -1, dtype=cp.int32)
    pos_m = cp.zeros(V, dtype=cp.float64)
    enter_time = cp.zeros(V, dtype=cp.float64)
    start_time = cp.full(V, -1.0, dtype=cp.float64)
    arrival_time = cp.full(V, -1.0, dtype=cp.float64)
    route_dist = cp.zeros(V, dtype=cp.float64)
    hold_until = cp.zeros(V, dtype=cp.float64)

    edge_count = cp.zeros(E, dtype=cp.int32)
    out_credit = cp.zeros(E, dtype=cp.float64)
    acc_count = cp.zeros(E, dtype=cp.float64)
    edge_exits = cp.zeros(E, dtype=cp.float64)
    espeed = cp.zeros(E, dtype=cp.float64)

    send_counter = cp.zeros(E, dtype=cp.int32)
    recv_counter = cp.zeros(E, dtype=cp.int32)
    dep_counter = cp.zeros(E, dtype=cp.int32)
    send_rank = cp.zeros(V, dtype=cp.int32)
    recv_rank = cp.zeros(V, dtype=cp.int32)
    next_edge = cp.zeros(V, dtype=cp.int32)
    first_edge = cp.zeros(V, dtype=cp.int32)
    dep_rank = cp.zeros(V, dtype=cp.int32)
    ecount_snap = cp.zeros(E, dtype=cp.int32)
    ecount_snap2 = cp.zeros(E, dtype=cp.int32)

    t_dev = cp.zeros(1, dtype=cp.float64)
    step_dev = cp.zeros(1, dtype=cp.int32)

    TPB = 256
    gV = (V + TPB - 1) // TPB
    gE = (E + TPB - 1) // TPB
    f64 = np.float64; i32 = np.int32

    def one_step():
        K["k_tick"]((1,), (1,), (np.float32(dt), step_dev, t_dev))
        K["k_edge_speed"]((gE,), (TPB,), (i32(E), edge_count, length, lanes, vmax,
                          f64(rho_jam), f64(args.min_speed), espeed))
        K["k_advance"]((gV,), (TPB,), (i32(V), t_dev, f64(dt), state, cur_edge, pos_m,
                       hold_until, espeed, length))
        K["k_credit"]((gE,), (TPB,), (i32(E), sat_cap, f64(dt), out_credit))
        send_counter.fill(0)
        K["k_send_ticket"]((gV,), (TPB,), (i32(V), state, cur_edge, send_counter, send_rank))
        recv_counter.fill(0)
        K["k_recv"]((gV,), (TPB,), (i32(V), state, cur_edge, cur_pos, veh_off, veh_len,
                   redges, out_credit, send_rank, recv_counter, next_edge, recv_rank))
        cp.copyto(ecount_snap, edge_count)
        K["k_apply_disch"]((gV,), (TPB,), (i32(V), t_dev, f64(jct_delay), state, cur_edge,
                          cur_pos, next_edge, recv_rank, jam_storage, ecount_snap, edge_count,
                          out_credit, edge_exits, arrival_time, route_dist, length, pos_m,
                          enter_time, hold_until))
        dep_counter.fill(0)
        K["k_dep_ticket"]((gV,), (TPB,), (i32(V), t_dev, state, veh_depart, veh_off, redges,
                         dep_counter, first_edge, dep_rank))
        cp.copyto(ecount_snap2, edge_count)
        K["k_dep_apply"]((gV,), (TPB,), (i32(V), t_dev, state, first_edge, dep_rank,
                        jam_storage, ecount_snap2, edge_count, cur_pos, cur_edge,
                        enter_time, start_time, pos_m))
        K["k_accum"]((gE,), (TPB,), (i32(E), edge_count, f64(dt), acc_count))

    use_graph = (not args.no_graph)
    cp.cuda.runtime.deviceSynchronize()

    if use_graph:
        # 워밍업 1스텝(컴파일/캐시) 후 캡처
        one_step()
        cp.cuda.runtime.deviceSynchronize()
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            stream.begin_capture()
            one_step()
            graph = stream.end_capture()
        # 워밍업이 step 1을 소비했으므로 상태 리셋 후 정확히 n_steps 실행
        # (간단화를 위해 리셋 대신 워밍업 1 + 캡처 1 = step 2개 이미 진행 → n_steps-2 추가 replay)
        log("CUDA Graph 캡처 완료")
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        for _ in range(max(n_steps - 2, 0)):
            graph.launch()
        cp.cuda.runtime.deviceSynchronize()
        elapsed = time.perf_counter() - t0
    else:
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            one_step()
        cp.cuda.runtime.deviceSynchronize()
        elapsed = time.perf_counter() - t0

    n_arr = int((state == 3).sum())
    mode = "graph" if use_graph else "no-graph"
    log(f"meso-GPU(branch-free,{mode}) 완료: {elapsed:.2f}s ({n_steps} steps), 도착={n_arr}/{V}")

    if args.edge_output_csv:
        t_acc = n_steps * dt
        mean_count = cp.asnumpy(acc_count) / max(t_acc, 1.0)
        ln = net.length_m.astype(np.float64); la = net.lanes.astype(np.float64); vm = net.vmax_mps.astype(np.float64)
        density = mean_count / np.maximum(ln*la, 1.0)
        speed = np.maximum(vm*(1.0 - density/rho_jam), 0.0)
        flow = cp.asnumpy(edge_exits) / max(t_acc, 1.0)
        out = Path(args.edge_output_csv); out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["edge_id","lanes","density_veh_per_m","speed_mps","flow_veh_per_s"])
            for i, eid in enumerate(net.edge_ids):
                w.writerow([eid, float(net.lanes[i]), float(density[i]), float(speed[i]), float(flow[i])])
        log(f"edge 집계 저장: {out}")


def main():
    p = argparse.ArgumentParser(description="Branch-free + CUDA Graphs mesoscopic GPU sim")
    p.add_argument("--net-file", default="./map_import/gangnam4_generated.net.xml")
    p.add_argument("--route-file", default="./map_import/gangnam4_generated.sorted.rou.xml")
    p.add_argument("--sim-time", type=float, default=3600.0)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--jam-density-per-lane", type=float, default=0.18)
    p.add_argument("--sat-flow-per-lane", type=float, default=0.5)
    p.add_argument("--vmax-scale", type=float, default=1.0)
    p.add_argument("--min-speed", type=float, default=0.3)
    p.add_argument("--junction-delay", type=float, default=0.0)
    p.add_argument("--max-vehicles", type=int, default=0)
    p.add_argument("--no-graph", action="store_true", help="CUDA Graph 미사용(branch-free만)")
    p.add_argument("--edge-output-csv", default="./gangnam4_cuda/results/meso_gpu_graph.edge.csv")
    args = p.parse_args()
    run_sim(args)


if __name__ == "__main__":
    main()
