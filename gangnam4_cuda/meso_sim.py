#!/usr/bin/env python3
"""
Mesoscopic 차량 단위 시뮬레이터 (time-stepped spatial-queue 모델).

CTM(밀도)과 달리 개별 차량을 edge→edge로 이동시킨다:
  - running: edge 통과 중. 밀도 기반 속도로 exit_time 결정.
  - queued:  edge 하류 끝 도달, 다음 edge 전이 대기.
  - 전이 제약: 송신 saturation capacity(veh/s) + 수신 storage space.
  - FIFO: 같은 edge 안에서 enter_time 순으로 우선 배출.

산출물:
  - 차량별 통행시간(travel time) → SUMO tripinfo와 직접 비교(신규 검증축)
  - edge별 시간평균 밀도/속도/유량 → 기존 edgedata 비교 유지
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from meso_common import MesoNet, load_meso_net, log, fail


def within_group_rank(group: np.ndarray, order: np.ndarray) -> np.ndarray:
    """각 원소가 자기 group 안에서 order 오름차순으로 몇 번째인지(0-base) 반환. 벡터화."""
    if group.size == 0:
        return np.zeros(0, dtype=np.int64)
    idx = np.lexsort((order, group))      # group 우선, 그 안에서 order
    sg = group[idx]
    pos = np.arange(sg.size)
    start = np.searchsorted(sg, sg, side="left")
    r = pos - start
    out = np.empty(sg.size, dtype=np.int64)
    out[idx] = r
    return out


def run_sim(args) -> None:
    net = load_meso_net(
        Path(args.net_file), Path(args.route_file),
        jam_density_per_lane=args.jam_density_per_lane,
        sat_flow_per_lane=args.sat_flow_per_lane,
        vmax_scale=args.vmax_scale,
        max_vehicles=args.max_vehicles,
    )
    V = net.n_veh
    E = net.n_edges
    if V == 0:
        fail("차량 0대")

    dt = float(args.dt)
    sim_time = float(args.sim_time)
    n_steps = int(round(sim_time / dt))
    rho_jam = float(args.jam_density_per_lane)

    # --- 차량 상태 ---
    STATE_PRE, STATE_RUN, STATE_QUEUE, STATE_DONE = 0, 1, 2, 3
    state = np.zeros(V, dtype=np.int8)
    cur_pos = np.full(V, -1, dtype=np.int32)       # route 내 현재 위치
    cur_edge = np.full(V, -1, dtype=np.int32)
    pos_m = np.zeros(V, dtype=np.float64)          # 현재 edge 진입 후 진행 거리(m). length 도달 시 queue
    enter_time = np.zeros(V, dtype=np.float64)     # 현재 edge 진입 시각(FIFO)
    hold_until = np.zeros(V, dtype=np.float64)     # 교차로 진입 지연: 이 시각까지 전진 보류
    start_time = np.full(V, -1.0, dtype=np.float64) # 네트워크 최초 진입
    arrival_time = np.full(V, -1.0, dtype=np.float64)
    route_dist = np.zeros(V, dtype=np.float64)     # 통과 거리 누적(평균속도용)

    edge_count = np.zeros(E, dtype=np.int32)
    out_credit = np.zeros(E, dtype=np.float64)     # 누적 유출 capacity(소수부 carryover)

    # edge 시간평균 누적
    acc_count = np.zeros(E, dtype=np.float64)      # Σ edge_count*dt → 평균 차량수
    edge_exits = np.zeros(E, dtype=np.float64)     # edge를 빠져나간 누적 차량 수(실측 flow용)
    t_acc = 0.0

    length = net.length_m.astype(np.float64)
    lanes = net.lanes.astype(np.float64)
    vmax = net.vmax_mps.astype(np.float64)
    jam_storage = net.jam_storage.astype(np.float64)
    sat_cap = net.sat_cap_per_s.astype(np.float64)
    veh_off = net.veh_route_off
    veh_len = net.veh_route_len
    redges = net.route_edges

    min_speed = float(args.min_speed)
    jct_delay = float(args.junction_delay)
    cong_coef = float(args.junction_cong_coef)
    max_jct_delay = float(args.max_junction_delay)

    def edge_speed_all() -> np.ndarray:
        """모든 edge의 현재 밀도 기반 Greenshields 속도(>= min_speed). [E]"""
        dens = edge_count / np.maximum(length * lanes, 1.0)
        v = vmax * (1.0 - dens / rho_jam)
        return np.maximum(v, min_speed)

    # 단일 차량 궤적 추적(검증용): id → 인덱스
    track_idx = -1
    track_rows: list = []
    if args.track_vehicle:
        id2i = {vid: i for i, vid in enumerate(net.veh_ids)}
        track_idx = id2i.get(args.track_vehicle, -1)
        if track_idx < 0:
            log(f"[track] 차량 id={args.track_vehicle} 없음 — 추적 생략")
        else:
            log(f"[track] 차량 {args.track_vehicle}(idx={track_idx}) 위치 추적 시작")

    dep_ptr = 0  # veh_depart 정렬 포인터
    t0 = time.perf_counter()
    t = 0.0
    for step in range(n_steps):
        t = step * dt

        # 0) 현재 밀도 기반 edge 속도(이번 스텝 동안 고정)
        espeed = edge_speed_all()

        # 1) running 차량 위치 전진 (현재 edge 밀도 속도로). length 도달 → queued.
        #    이렇게 하면 진입 후 edge가 막히면 차량도 함께 느려짐(stop-and-go).
        run_mask = state == STATE_RUN
        if run_mask.any():
            rv = np.flatnonzero(run_mask)
            # 교차로 진입 지연(hold_until) 경과한 차량만 전진
            active = rv[hold_until[rv] <= t]
            pos_m[active] += espeed[cur_edge[active]] * dt
            reached = active[pos_m[active] >= length[cur_edge[active]]]
            state[reached] = STATE_QUEUE

        # 2) 유출 capacity 누적
        out_credit += sat_cap * dt

        # 3) Discharge: queued 차량 전이
        q = np.flatnonzero(state == STATE_QUEUE)
        if q.size > 0:
            ce = cur_edge[q]
            npos = cur_pos[q] + 1
            rlen = veh_len[q]
            is_exit = npos >= rlen                       # 네트워크 이탈(도착)
            ne = np.full(q.size, -1, dtype=np.int64)
            inq = ~is_exit
            if inq.any():
                ne[inq] = redges[veh_off[q[inq]] + npos[inq]]

            # (a) 송신 제약: edge별 out_credit(정수부)만큼, FIFO(enter_time)
            send_rank = within_group_rank(ce, enter_time[q])
            send_cap = np.floor(out_credit[ce]).astype(np.int64)
            elig_send = send_rank < send_cap

            # (b) 수신 제약: next_edge별 잔여 space, FIFO. sink(-1)은 무제한.
            space = np.floor(jam_storage - edge_count).astype(np.int64)
            cand = elig_send.copy()
            # 네트워크-내 이동만 수신 제약 검사
            move_in = cand & (~is_exit)
            if move_in.any():
                gi = ne[move_in]
                recv_rank = within_group_rank(gi.astype(np.int64), enter_time[q][move_in])
                recv_ok = recv_rank < np.maximum(space[gi], 0)
                tmp = np.flatnonzero(move_in)
                cand[tmp[~recv_ok]] = False

            movers = np.flatnonzero(cand)
            if movers.size > 0:
                gv = q[movers]                 # 글로벌 차량 인덱스
                gce = ce[movers]
                gne = ne[movers]
                g_exit = is_exit[movers]

                # 송신 edge에서 제거 + 실측 throughput 카운트
                np.add.at(edge_count, gce, -1)
                np.add.at(out_credit, gce, -1.0)  # 1대 배출당 credit 1 소모
                np.add.at(edge_exits, gce, 1.0)   # edge를 떠난 차량(실측 flow)

                # 도착 처리
                arr = gv[g_exit]
                arrival_time[arr] = t
                state[arr] = STATE_DONE
                cur_edge[arr] = -1

                # 네트워크-내 이동 처리
                mv = gv[~g_exit]
                mv_ne = gne[~g_exit]
                if mv.size > 0:
                    route_dist[mv] += length[gce[~g_exit]]  # 직전 edge 길이 통과
                    cur_pos[mv] += 1
                    cur_edge[mv] = mv_ne.astype(np.int32)
                    enter_time[mv] = t
                    pos_m[mv] = 0.0
                    # 교차로 통과 지연: base + 혼잡비례(목적지 점유율 occ의 Webster-overflow형)
                    if cong_coef > 0.0:
                        occ = np.clip(edge_count[mv_ne] / np.maximum(jam_storage[mv_ne], 1e-9), 0.0, 0.99)
                        delay = jct_delay + cong_coef * occ / (1.0 - occ)
                        np.minimum(delay, max_jct_delay, out=delay)
                        hold_until[mv] = t + delay
                    else:
                        hold_until[mv] = t + jct_delay
                    np.add.at(edge_count, mv_ne, 1)
                    state[mv] = STATE_RUN

        # 4) Departures: depart<=t 인 PRE 차량을 첫 edge에 투입(잔여 space 한도)
        while dep_ptr < V and net.veh_depart[dep_ptr] <= t:
            dep_ptr += 1
        # PRE & depart<=t 후보
        cand_dep = np.flatnonzero((state == STATE_PRE) & (net.veh_depart <= t))
        if cand_dep.size > 0:
            e0 = redges[veh_off[cand_dep]]                 # 각 차량 첫 edge
            # FIFO by depart, edge별 잔여 space
            drank = within_group_rank(e0.astype(np.int64), net.veh_depart[cand_dep].astype(np.float64))
            space0 = np.floor(jam_storage - edge_count).astype(np.int64)
            ok = drank < np.maximum(space0[e0], 0)
            ins = cand_dep[ok]
            if ins.size > 0:
                ie = e0[ok]
                state[ins] = STATE_RUN
                cur_pos[ins] = 0
                cur_edge[ins] = ie.astype(np.int32)
                enter_time[ins] = t
                start_time[ins] = t
                pos_m[ins] = 0.0
                np.add.at(edge_count, ie, 1)

        # 5) edge 시간평균 누적
        acc_count += edge_count * dt
        t_acc += dt

        # (검증) 추적 차량의 위치 기록: 어느 edge의 몇 m 지점 + 누적거리 + 상태
        if track_idx >= 0:
            st = int(state[track_idx])
            ei = int(cur_edge[track_idx])
            cum = float(route_dist[track_idx] + (pos_m[track_idx] if st in (STATE_RUN, STATE_QUEUE) else 0.0))
            track_rows.append([
                f"{t:.1f}", {0: "PRE", 1: "RUN", 2: "QUEUE", 3: "DONE"}[st],
                (net.edge_ids[ei] if ei >= 0 else "-"),
                f"{float(pos_m[track_idx]):.2f}", f"{cum:.2f}",
                f"{int(cur_pos[track_idx])}", f"{int(veh_len[track_idx])}",
            ])

        if (step + 1) % args.log_interval == 0:
            running = int((state == STATE_RUN).sum())
            queued = int((state == STATE_QUEUE).sum())
            done = int((state == STATE_DONE).sum())
            log(f"step={step+1}/{n_steps} t={t:.0f}s running={running} queued={queued} "
                f"arrived={done} pending={V-running-queued-done}")

    elapsed = time.perf_counter() - t0
    n_arr = int((state == STATE_DONE).sum())
    log(f"meso 시뮬레이션 완료: {elapsed:.2f}s ({n_steps} steps), 도착={n_arr}/{V}")

    # (검증) 추적 차량 궤적 저장
    if track_idx >= 0 and args.track_output:
        out = Path(args.track_output); out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "state", "edge_id", "pos_m_on_edge", "cum_dist_m", "route_idx", "route_len"])
            w.writerows(track_rows)
        moved = [r for r in track_rows if r[1] in ("RUN", "QUEUE")]
        log(f"[track] 궤적 저장: {out} ({len(track_rows)} steps, 이동구간 {len(moved)} steps)")

    # --- 차량별 통행시간 출력 ---
    if args.trip_output_csv:
        tt = arrival_time - start_time
        valid = (state == STATE_DONE) & (start_time >= 0)
        out = Path(args.trip_output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["veh_id", "depart", "start", "arrival", "travel_time_s", "route_dist_m", "mean_speed_mps"])
            for v in np.flatnonzero(valid):
                d = float(route_dist[v]); dur = float(tt[v])
                w.writerow([net.veh_ids[v], float(net.veh_depart[v]), float(start_time[v]),
                            float(arrival_time[v]), dur, d, (d / dur if dur > 0 else 0.0)])
        log(f"차량 통행시간 저장: {out} (완주 차량 {int(valid.sum())}대)")
        if valid.any():
            durs = tt[valid]
            log(f"통행시간 통계: mean={durs.mean():.1f}s median={np.median(durs):.1f}s p95={np.percentile(durs,95):.1f}s")

    # --- edge 시간평균 → edgedata 비교용 CSV ---
    if args.edge_output_csv and t_acc > 0:
        mean_count = acc_count / t_acc
        density = mean_count / np.maximum(length * lanes, 1.0)       # edge density veh/m
        speed = np.maximum(vmax * (1.0 - (density / rho_jam)), 0.0)
        # 실측 flow: edge를 빠져나간 차량수 / sim_time (veh/s)
        flow = edge_exits / max(t_acc, 1.0)
        out = Path(args.edge_output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["edge_id", "lanes", "density_veh_per_m", "speed_mps", "flow_veh_per_s"])
            for i, eid in enumerate(net.edge_ids):
                w.writerow([eid, float(net.lanes[i]), float(density[i]), float(speed[i]), float(flow[i])])
        log(f"edge 집계 저장: {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Mesoscopic vehicle-level traffic simulator")
    p.add_argument("--net-file", default="./map_import/gangnam4_generated.net.xml")
    p.add_argument("--route-file", default="./map_import/gangnam4_generated.sorted.rou.xml")
    p.add_argument("--sim-time", type=float, default=3600.0)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--jam-density-per-lane", type=float, default=0.18)
    p.add_argument("--sat-flow-per-lane", type=float, default=0.5, help="차로당 saturation flow(veh/s), ~1800veh/h")
    p.add_argument("--vmax-scale", type=float, default=1.0)
    p.add_argument("--min-speed", type=float, default=0.3,
                   help="혼잡 edge 최소 통과속도(m/s). 낮을수록 jam에서 더 오래 정체→SUMO 통행시간에 근접. 0.3=기본")
    p.add_argument("--junction-delay", type=float, default=0.0,
                   help="edge 전이(교차로 통과) 고정 base 지연(s). 신호/양보 대기 근사")
    p.add_argument("--junction-cong-coef", type=float, default=0.0,
                   help="혼잡비례 지연 계수. delay=base+coef*occ/(1-occ) (occ=목적지 점유율). Webster-overflow형")
    p.add_argument("--max-junction-delay", type=float, default=120.0,
                   help="혼잡비례 지연 상한(s)")
    p.add_argument("--max-vehicles", type=int, default=0, help=">0이면 출발순 앞쪽 N대만(테스트)")
    p.add_argument("--track-vehicle", default="", help="검증용: 이 차량 id의 위치를 매 스텝 기록")
    p.add_argument("--track-output", default="./gangnam4_cuda/results/track.csv", help="추적 궤적 CSV 경로")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=600)
    p.add_argument("--trip-output-csv", default="./gangnam4_cuda/results/meso_trips.csv")
    p.add_argument("--edge-output-csv", default="./gangnam4_cuda/results/meso_state.edge.csv")
    args = p.parse_args()
    run_sim(args)


if __name__ == "__main__":
    main()
