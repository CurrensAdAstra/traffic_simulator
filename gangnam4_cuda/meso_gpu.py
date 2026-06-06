#!/usr/bin/env python3
"""
GPU 메소스코픽 차량 단위 시뮬레이터 — meso_sim.py의 cupy 포팅.

CPU 버전과 동일한 time-stepped spatial-queue 모델(running→queue→discharge, FIFO +
saturation/storage 제약, stop-and-go 위치전진, junction-delay)을 cupy로 디바이스에서 실행.
  np.add.at        → cupyx.scatter_add
  np.lexsort       → cp.lexsort  (within_group_rank)
  상태/네트워크 배열 → 디바이스 상주(스텝 루프는 host 제어, 연산은 GPU)

대규모(차량 100k–1M)에서 CPU meso 대비 이득을 노린다(스텝당 GPU 병렬).
출력(통행시간/edge 집계 CSV)은 CPU 버전과 동일 스키마.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from meso_common import load_meso_net, log, fail


def within_group_rank_gpu(cp, group, order):
    """각 원소가 자기 group 안에서 order 오름차순으로 몇 번째(0-base)인지. cupy."""
    if group.size == 0:
        return cp.zeros(0, dtype=cp.int64)
    idx = cp.lexsort(cp.stack([order, group]))   # group 우선(마지막 행), order 보조
    sg = group[idx]
    pos = cp.arange(sg.size)
    start = cp.searchsorted(sg, sg, side="left")
    r = pos - start
    out = cp.empty(sg.size, dtype=cp.int64)
    out[idx] = r
    return out


def build_ticket_kernel(cp):
    """그룹 내 순위를 atomicAdd로 부여(정렬 불필요). rank[i] = (그 그룹에서 i보다 먼저 처리된 수).
    순서는 비결정적이지만 그룹당 capacity 배분에는 충분(메소 근사). lexsort O(NlogN) 제거."""
    return cp.RawKernel(r'''
    extern "C" __global__
    void ticket(const int n, const int* group, int* counter, int* rank){
        int i = blockDim.x*blockIdx.x + threadIdx.x;
        if (i >= n) return;
        rank[i] = atomicAdd(&counter[group[i]], 1);
    }
    ''', "ticket")


def run_sim(args) -> None:
    try:
        import cupy as cp  # type: ignore
        from cupyx import scatter_add  # type: ignore
    except Exception as e:
        fail(f"cupy import 실패: {e}")

    net = load_meso_net(
        Path(args.net_file), Path(args.route_file),
        jam_density_per_lane=args.jam_density_per_lane,
        sat_flow_per_lane=args.sat_flow_per_lane,
        vmax_scale=args.vmax_scale, max_vehicles=args.max_vehicles,
    )
    V, E = net.n_veh, net.n_edges
    if V == 0:
        fail("차량 0대")
    dt = float(args.dt)
    n_steps = int(round(args.sim_time / dt))
    rho_jam = float(args.jam_density_per_lane)
    min_speed = float(args.min_speed)
    jct_delay = float(args.junction_delay)

    STATE_PRE, STATE_RUN, STATE_QUEUE, STATE_DONE = 0, 1, 2, 3

    # 네트워크 배열(디바이스)
    length = cp.asarray(net.length_m, dtype=cp.float64)
    lanes = cp.asarray(net.lanes, dtype=cp.float64)
    vmax = cp.asarray(net.vmax_mps, dtype=cp.float64)
    jam_storage = cp.asarray(net.jam_storage, dtype=cp.float64)
    sat_cap = cp.asarray(net.sat_cap_per_s, dtype=cp.float64)
    veh_off = cp.asarray(net.veh_route_off)
    veh_len = cp.asarray(net.veh_route_len)
    redges = cp.asarray(net.route_edges)
    veh_depart = cp.asarray(net.veh_depart, dtype=cp.float64)

    # 차량 상태(디바이스)
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
    t_acc = 0.0

    def edge_speed_all():
        dens = edge_count / cp.maximum(length * lanes, 1.0)
        return cp.maximum(vmax * (1.0 - dens / rho_jam), min_speed)

    # 그룹 내 순위: sort(lexsort, 정확 FIFO) 또는 ticket(atomic, 정렬無 근사)
    rank_mode = args.rank_mode
    ticket_kernel = build_ticket_kernel(cp) if rank_mode == "ticket" else None
    _tcounter = cp.zeros(E, dtype=cp.int32)  # ticket용 그룹 카운터(edge별)

    def group_rank(group_i64, order_f64, group_for_ticket_i32):
        """rank_mode에 따라 그룹 내 순위 반환(int64). ticket은 order 무시(비FIFO 근사)."""
        if rank_mode == "sort":
            return within_group_rank_gpu(cp, group_i64, order_f64)
        n = group_for_ticket_i32.size
        if n == 0:
            return cp.zeros(0, dtype=cp.int64)
        _tcounter.fill(0)
        rank = cp.empty(n, dtype=cp.int32)
        threads = 256; blocks = (n + threads - 1) // threads
        ticket_kernel((blocks,), (threads,), (np.int32(n), group_for_ticket_i32, _tcounter, rank))
        return rank.astype(cp.int64)

    # 단일 차량 궤적 추적(검증용)
    track_idx = -1
    track_rows: list = []
    if args.track_vehicle:
        id2i = {vid: i for i, vid in enumerate(net.veh_ids)}
        track_idx = id2i.get(args.track_vehicle, -1)
        if track_idx >= 0:
            log(f"[track] 차량 {args.track_vehicle}(idx={track_idx}) 위치 추적 시작")

    # --- 프로파일링: 섹션 경계마다 mark() (sync 후 직전 mark 이후 경과시간 누적) ---
    do_prof = args.profile
    prof: dict = {}
    _last = [0.0]
    def mark(name: str):
        if not do_prof:
            return
        cp.cuda.runtime.deviceSynchronize()
        now = time.perf_counter()
        prof[name] = prof.get(name, 0.0) + (now - _last[0])
        _last[0] = now

    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    for step in range(n_steps):
        t = step * dt
        if do_prof:
            cp.cuda.runtime.deviceSynchronize(); _last[0] = time.perf_counter()
        espeed = edge_speed_all()

        # 1) running 위치 전진(hold_until 경과분만), length 도달 → queue
        run_mask = state == STATE_RUN
        rv = cp.flatnonzero(run_mask)
        if rv.size > 0:
            active = rv[hold_until[rv] <= t]
            if active.size > 0:
                pos_m[active] += espeed[cur_edge[active]] * dt
                reached = active[pos_m[active] >= length[cur_edge[active]]]
                state[reached] = STATE_QUEUE
        mark("advance")

        # 2) 유출 capacity 누적
        out_credit += sat_cap * dt

        # 3) Discharge
        q = cp.flatnonzero(state == STATE_QUEUE)
        if q.size > 0:
            ce = cur_edge[q]
            npos = cur_pos[q] + 1
            rlen = veh_len[q]
            is_exit = npos >= rlen
            ne = cp.full(q.size, -1, dtype=cp.int64)
            inq = ~is_exit
            if bool(inq.any()):
                ne[inq] = redges[veh_off[q[inq]] + npos[inq]]
            mark("disch_setup")

            ce32 = ce.astype(cp.int32)
            send_rank = group_rank(ce.astype(cp.int64), enter_time[q], ce32)
            send_cap = cp.floor(out_credit[ce]).astype(cp.int64)
            elig_send = send_rank < send_cap
            mark("sort_send")

            space = cp.floor(jam_storage - edge_count).astype(cp.int64)
            cand = elig_send.copy()
            move_in = cand & (~is_exit)
            if bool(move_in.any()):
                gi = ne[move_in]
                recv_rank = group_rank(gi.astype(cp.int64), enter_time[q][move_in], gi.astype(cp.int32))
                recv_ok = recv_rank < cp.maximum(space[gi], 0)
                tmp = cp.flatnonzero(move_in)
                cand[tmp[~recv_ok]] = False
            mark("sort_recv")

            movers = cp.flatnonzero(cand)
            if movers.size > 0:
                gv = q[movers]; gce = ce[movers]; gne = ne[movers]; g_exit = is_exit[movers]
                scatter_add(edge_count, gce, cp.int32(-1))
                scatter_add(out_credit, gce, -1.0)
                scatter_add(edge_exits, gce, 1.0)

                arr = gv[g_exit]
                arrival_time[arr] = t
                state[arr] = STATE_DONE
                cur_edge[arr] = -1

                nx = ~g_exit
                mv = gv[nx]; mv_ne = gne[nx]
                if mv.size > 0:
                    route_dist[mv] += length[gce[nx]]
                    cur_pos[mv] += 1
                    cur_edge[mv] = mv_ne.astype(cp.int32)
                    enter_time[mv] = t
                    pos_m[mv] = 0.0
                    hold_until[mv] = t + jct_delay
                    scatter_add(edge_count, mv_ne, cp.int32(1))
                    state[mv] = STATE_RUN
            mark("disch_apply")

        # 4) Departures: PRE & depart<=t, 첫 edge space 한도
        cand_dep = cp.flatnonzero((state == STATE_PRE) & (veh_depart <= t))
        if cand_dep.size > 0:
            e0 = redges[veh_off[cand_dep]]
            drank = group_rank(e0.astype(cp.int64), veh_depart[cand_dep], e0.astype(cp.int32))
            space0 = cp.floor(jam_storage - edge_count).astype(cp.int64)
            ok = drank < cp.maximum(space0[e0], 0)
            ins = cand_dep[ok]
            if ins.size > 0:
                ie = e0[ok]
                state[ins] = STATE_RUN
                cur_pos[ins] = 0
                cur_edge[ins] = ie.astype(cp.int32)
                enter_time[ins] = t
                start_time[ins] = t
                pos_m[ins] = 0.0
                scatter_add(edge_count, ie, cp.int32(1))
        mark("departures")

        # 5) edge 시간평균 누적
        acc_count += edge_count.astype(cp.float64) * dt
        t_acc += dt
        mark("accumulate")

        # (검증) 추적 차량 위치 — device 스칼라 1개씩 host로
        if track_idx >= 0:
            st = int(state[track_idx]); ei = int(cur_edge[track_idx])
            pm = float(pos_m[track_idx])
            cum = float(route_dist[track_idx]) + (pm if st in (1, 2) else 0.0)
            track_rows.append([
                f"{t:.1f}", {0: "PRE", 1: "RUN", 2: "QUEUE", 3: "DONE"}[st],
                (net.edge_ids[ei] if ei >= 0 else "-"),
                f"{pm:.2f}", f"{cum:.2f}",
                f"{int(cur_pos[track_idx])}", f"{int(veh_len[track_idx])}",
            ])

        if (step + 1) % args.log_interval == 0:
            running = int((state == STATE_RUN).sum())
            queued = int((state == STATE_QUEUE).sum())
            done = int((state == STATE_DONE).sum())
            log(f"step={step+1}/{n_steps} t={t:.0f}s running={running} queued={queued} arrived={done}")

    cp.cuda.runtime.deviceSynchronize()
    elapsed = time.perf_counter() - t0
    n_arr = int((state == STATE_DONE).sum())
    log(f"meso-GPU 시뮬레이션 완료: {elapsed:.2f}s ({n_steps} steps), 도착={n_arr}/{V}")

    if do_prof and prof:
        tot = sum(prof.values())
        log("=" * 56)
        log(f"[PROFILE] 섹션별 GPU+host 시간 (instrumented 합 {tot:.2f}s, 관측자효과로 부풀려짐)")
        order = ["advance", "disch_setup", "sort_send", "sort_recv", "disch_apply", "departures", "accumulate"]
        for k in order + [x for x in prof if x not in order]:
            if k in prof:
                log(f"  {k:14s} {prof[k]:8.2f}s  {100*prof[k]/tot:5.1f}%")
        sort_pct = 100 * (prof.get("sort_send", 0) + prof.get("sort_recv", 0) +
                          prof.get("departures", 0)) / tot
        log(f"  → 정렬(lexsort) 계열 합: ~{sort_pct:.0f}%  (send+recv+departures 내 rank)")
        log("=" * 56)

    if track_idx >= 0 and args.track_output:
        out = Path(args.track_output); out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "state", "edge_id", "pos_m_on_edge", "cum_dist_m", "route_idx", "route_len"])
            w.writerows(track_rows)
        log(f"[track] 궤적 저장: {out} ({len(track_rows)} steps)")

    # 결과 host로
    state_h = cp.asnumpy(state)
    start_h = cp.asnumpy(start_time)
    arr_h = cp.asnumpy(arrival_time)
    dist_h = cp.asnumpy(route_dist)

    if args.trip_output_csv:
        tt = arr_h - start_h
        valid = (state_h == STATE_DONE) & (start_h >= 0)
        out = Path(args.trip_output_csv); out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["veh_id", "depart", "start", "arrival", "travel_time_s", "route_dist_m", "mean_speed_mps"])
            for v in np.flatnonzero(valid):
                d = float(dist_h[v]); dur = float(tt[v])
                w.writerow([net.veh_ids[v], float(net.veh_depart[v]), float(start_h[v]),
                            float(arr_h[v]), dur, d, (d / dur if dur > 0 else 0.0)])
        log(f"차량 통행시간 저장: {out} (완주 {int(valid.sum())}대)")
        if valid.any():
            durs = tt[valid]
            log(f"통행시간 통계: mean={durs.mean():.1f}s median={np.median(durs):.1f}s p95={np.percentile(durs,95):.1f}s")

    if args.edge_output_csv and t_acc > 0:
        mean_count = cp.asnumpy(acc_count) / t_acc
        ln = net.length_m.astype(np.float64); la = net.lanes.astype(np.float64); vm = net.vmax_mps.astype(np.float64)
        density = mean_count / np.maximum(ln * la, 1.0)
        speed = np.maximum(vm * (1.0 - density / rho_jam), 0.0)
        flow = cp.asnumpy(edge_exits) / max(t_acc, 1.0)
        out = Path(args.edge_output_csv); out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["edge_id", "lanes", "density_veh_per_m", "speed_mps", "flow_veh_per_s"])
            for i, eid in enumerate(net.edge_ids):
                w.writerow([eid, float(net.lanes[i]), float(density[i]), float(speed[i]), float(flow[i])])
        log(f"edge 집계 저장: {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="GPU mesoscopic vehicle-level traffic simulator")
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
    p.add_argument("--track-vehicle", default="", help="검증용: 이 차량 id 위치를 매 스텝 기록")
    p.add_argument("--track-output", default="./gangnam4_cuda/results/track_gpu.csv")
    p.add_argument("--profile", action="store_true", help="스텝 섹션별 시간 분해(sync 포함, 절대값은 관측자효과로 부풀려짐)")
    p.add_argument("--rank-mode", default="sort", choices=["sort", "ticket"],
                   help="그룹내 순위: sort(lexsort, 정확 FIFO) | ticket(atomic, 정렬無 근사·고속)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=600)
    p.add_argument("--trip-output-csv", default="./gangnam4_cuda/results/meso_gpu_trips.csv")
    p.add_argument("--edge-output-csv", default="./gangnam4_cuda/results/meso_gpu_state.edge.csv")
    args = p.parse_args()
    run_sim(args)


if __name__ == "__main__":
    main()
