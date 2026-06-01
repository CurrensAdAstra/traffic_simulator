#!/usr/bin/env python3
"""
Multi-cell CTM CPU 엔진 — lane_cpu_simulator_mt의 cell-단위 확장.

핵심 차이:
 - 시뮬레이션 단위 = cell (lane을 N개로 세분화한 공간 격자)
 - intra-lane(같은 lane의 cell[i]→[i+1])과 inter-lane(lane 경계 통과) 두 종류 연결
 - lateral lane-change는 일단 생략(다음 단계에서 lane-level로 추가 가능)
 - vectorized NumPy로 lane CTM과 동일한 수식 적용
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from lane_common import (
    DIR_L, DIR_S, DIR_T,
    build_demand_and_target,
    load_lane_net,
    log,
    fail,
)
from cell_common import (
    CellNet,
    build_cell_net,
    cells_to_lanes,
    aggregate_cells_to_edges,
)


def ctm_step_cell(
    cnet: CellNet,
    rho: np.ndarray,
    rho_jam: float,
    length_eff: np.ndarray,
    dt: float,
    source_demand_cell: np.ndarray,
    conn_split: np.ndarray,
    jc_cap=None,  # 미사용(현재 cell 엔진은 junction cap 미적용)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized NumPy CTM 한 스텝 (cell 단위). 측면 차선변경 없음."""
    vmax = cnet.vmax
    rho_c = rho_jam * 0.5
    q_max = vmax * (rho_jam * 0.25)
    q = rho * vmax * (1.0 - rho / rho_jam)
    S = np.where(rho <= rho_c, q, q_max)
    R = np.where(rho >= rho_c, q, q_max)

    # 연결별 demand (사전 계산된 conn_split = 원본 * route 보정 * HCM penalty)
    D = S[cnet.conn_src] * conn_split

    # priority/yield 머지: 수신측에서 major가 먼저, minor는 잔여만
    pri = cnet.conn_priority.astype(bool)
    D_major = np.where(pri, D, 0.0)
    D_minor = np.where(pri, 0.0, D)

    n = cnet.n_cells
    tot_major = np.zeros(n, dtype=D.dtype)
    tot_minor = np.zeros(n, dtype=D.dtype)
    np.add.at(tot_major, cnet.conn_dst, D_major)
    np.add.at(tot_minor, cnet.conn_dst, D_minor)

    scale_major = np.minimum(1.0,
        np.divide(R, tot_major, out=np.ones_like(R), where=tot_major > 0.0))
    served_major = tot_major * scale_major
    residual = np.maximum(R - served_major, 0.0)
    scale_minor = np.minimum(1.0,
        np.divide(residual, tot_minor, out=np.ones_like(residual), where=tot_minor > 0.0))
    scale_per_conn = np.where(pri, scale_major[cnet.conn_dst], scale_minor[cnet.conn_dst])
    Q = D * scale_per_conn

    inflow = np.zeros(n, dtype=D.dtype)
    outflow = np.zeros(n, dtype=D.dtype)
    np.add.at(inflow, cnet.conn_dst, Q)
    np.add.at(outflow, cnet.conn_src, Q)

    # 진입 cell(=lane head 중 no_incoming) 에만 외부 inflow
    if source_demand_cell is not None:
        inflow = inflow + source_demand_cell

    # sink cell(하류 없음 = 마지막 lane들의 마지막 cell 중 lane이 no_outgoing) 자유 유출
    # (정적 마스크 — caller가 precompute)
    # → caller는 outflow += S * no_outgoing_mask 를 외부에서 더해줘도 되지만
    #    여기선 매 스텝 결정되는 S에 의존하므로 함수 내부에서 처리 권장. 단순화 위해 caller로 위임.

    rho_long = rho + dt * (inflow - outflow) / length_eff
    np.clip(rho_long, 0.0, rho_jam, out=rho_long)

    # 측면(lane-change) 생략 — rho_long이 곧 rho_next
    rho_next = rho_long
    speed = np.maximum(vmax * (1.0 - rho_next / rho_jam), 0.0)
    flow_next = rho_next * speed
    return rho_next, speed, flow_next


def run_sim(args) -> None:
    lane_net = load_lane_net(Path(args.net_file))
    # vmax-scale 적용 (lane_net level에 먼저)
    if args.vmax_scale != 1.0:
        lane_net.vmax_mps = (lane_net.vmax_mps * np.float32(args.vmax_scale)).astype(np.float32)
        log(f"vmax-scale={args.vmax_scale} 적용")

    cnet = build_cell_net(lane_net, target_cell_length=float(args.cell_length))
    n = cnet.n_cells
    rho_jam = float(args.jam_density_per_lane)

    # 수요/회전 보정 — lane 단위로 받아서 cell로 변환
    source_demand_lane, target_share_lane, conn_split_cal_lane, veh_n = build_demand_and_target(
        lane_net, Path(args.net_file),
        Path(args.route_file) if args.route_file else None,
        args.sim_duration, args.source_demand,
    )

    # HCM movement penalty — lane-level conn에만 적용 (intra는 항상 1.0)
    if args.major_left_factor != 1.0 or args.minor_factor != 1.0:
        mvf = np.ones(lane_net.n_conn, dtype=np.float32)
        pri_b = lane_net.conn_priority.astype(bool)
        is_left_or_u = (lane_net.conn_dir == DIR_L) | (lane_net.conn_dir == DIR_T)
        mvf[pri_b & is_left_or_u] = float(args.major_left_factor)
        mvf[~pri_b] = float(args.minor_factor)
        conn_split_cal_lane = (conn_split_cal_lane * mvf).astype(np.float32)
        log(f"HCM movement penalty: ml={args.major_left_factor}, m={args.minor_factor}")

    # cell-level conn_split: intra는 1.0, inter는 conn_split_cal_lane
    intra_n = cnet.n_conn - lane_net.n_conn
    conn_split_cell = np.concatenate([
        np.ones(intra_n, dtype=np.float32),
        conn_split_cal_lane.astype(np.float32),
    ])

    # source_demand: lane source를 차선의 모든 cell에 균등 분배 (lane head-only이면 head 병목 발생).
    # lane의 source veh/s를 lane 전체 길이에 분산 → cell당 veh/s = source * (cell.length/lane.length)
    source_demand_cell = np.zeros(n, dtype=np.float32)
    for li in range(lane_net.n_lanes):
        s = int(cnet.lane_cell_first[li])
        e = s + int(cnet.lane_cell_count[li])
        lane_len = float(lane_net.length_m[li])
        if lane_len > 0:
            cell_lens = cnet.length[s:e]
            source_demand_cell[s:e] = source_demand_lane[li] * (cell_lens / lane_len).astype(np.float32)

    # no_outgoing mask: cell이 outgoing connection이 없으면 sink (외부로 자유 유출)
    out_count = np.bincount(cnet.conn_src, minlength=n)
    no_outgoing_cell = (out_count == 0).astype(np.int8)
    log(f"cell 수={n}, intra_conn={intra_n}, inter_conn={lane_net.n_conn}, "
        f"진입 cell={int(cnet.no_incoming_mask.sum())}, 출구 cell={int(no_outgoing_cell.sum())}")

    # --- sim-time → steps 정렬 ---
    steps = args.steps
    if args.sim_time and args.sim_time > 0:
        steps = max(1, int(round(args.sim_time / args.dt)))
        log(f"sim_time={args.sim_time}s 적용: steps={steps} (dt={args.dt})")

    # CFL-안정 effective length
    length_eff = np.maximum(cnet.length, np.float32(args.dt) * cnet.vmax).astype(np.float32)

    # 초기 상태
    rng = np.random.default_rng(args.seed)
    rho = (args.init_density * rng.uniform(0.7, 1.3, size=n)).astype(np.float32)
    rho = np.clip(rho, 0.0, rho_jam * 0.95)

    rho_acc = np.zeros(n, dtype=np.float64) if args.time_average else None
    speed_acc = np.zeros(n, dtype=np.float64) if args.time_average else None
    flow_acc = np.zeros(n, dtype=np.float64) if args.time_average else None
    t_acc = 0.0

    def step_once():
        nonlocal rho
        # ctm_step_cell 내부에서 sink 처리는 외부 hook이 필요 — 여기서 wrapper로 처리:
        # 함수가 outflow에 no_outgoing 처리를 넣지 않으므로 여기서 보정.
        # 우회: ctm_step_cell의 마지막에 외부 outflow 더하기 위해 별도 처리.
        # 간단하게: ctm_step_cell를 수정하기보다 step 후 sink_drain을 별도 적용하면 복잡해짐.
        # 더 간단한 방법: 위 ctm_step_cell에서 sink 처리를 직접 포함하도록 인자로 전달.
        pass

    # (re-)간소화: 함수 본문에서 sink 처리를 inline으로 처리하기 위해 step 코드를 여기서 펼침
    def ctm_one_step(rho_in: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vmax = cnet.vmax
        rho_c = rho_jam * 0.5
        q_max = vmax * (rho_jam * 0.25)
        q = rho_in * vmax * (1.0 - rho_in / rho_jam)
        S = np.where(rho_in <= rho_c, q, q_max)
        R = np.where(rho_in >= rho_c, q, q_max)

        D = S[cnet.conn_src] * conn_split_cell
        pri = cnet.conn_priority.astype(bool)
        D_major = np.where(pri, D, 0.0)
        D_minor = np.where(pri, 0.0, D)

        tot_major = np.zeros(n, dtype=D.dtype)
        tot_minor = np.zeros(n, dtype=D.dtype)
        np.add.at(tot_major, cnet.conn_dst, D_major)
        np.add.at(tot_minor, cnet.conn_dst, D_minor)

        scale_major = np.minimum(1.0,
            np.divide(R, tot_major, out=np.ones_like(R), where=tot_major > 0.0))
        served_major = tot_major * scale_major
        residual = np.maximum(R - served_major, 0.0)
        scale_minor = np.minimum(1.0,
            np.divide(residual, tot_minor, out=np.ones_like(residual), where=tot_minor > 0.0))
        scale_per_conn = np.where(pri, scale_major[cnet.conn_dst], scale_minor[cnet.conn_dst])
        Q = D * scale_per_conn

        inflow = np.zeros(n, dtype=D.dtype)
        outflow = np.zeros(n, dtype=D.dtype)
        np.add.at(inflow, cnet.conn_dst, Q)
        np.add.at(outflow, cnet.conn_src, Q)
        # 진입(외부 inflow) + 출구(외부 outflow = S 만큼 drain)
        inflow = inflow + source_demand_cell
        outflow = outflow + S * no_outgoing_cell

        rho_long = rho_in + dt_f32 * (inflow - outflow) / length_eff
        np.clip(rho_long, 0.0, rho_jam, out=rho_long)
        speed = np.maximum(vmax * (1.0 - rho_long / rho_jam), 0.0)
        flow_next = rho_long * speed
        return rho_long, speed, flow_next

    dt_f32 = np.float32(args.dt)
    # 워밍업 1 스텝 (누적 X)
    rho, speed, flow = ctm_one_step(rho)

    t0 = time.perf_counter()
    for step in range(steps):
        rho, speed, flow = ctm_one_step(rho)
        if args.time_average:
            rho_acc += rho * args.dt
            speed_acc += speed * args.dt
            flow_acc += flow * args.dt
            t_acc += args.dt

        if (step + 1) % args.log_interval == 0:
            log(f"step={step+1}/{steps} mean_speed={float(np.mean(speed)):.3f} m/s "
                f"mean_density={float(np.mean(rho)):.6f} veh/m")

    elapsed = time.perf_counter() - t0
    log(f"CPU(cell,ctm) 시뮬레이션 완료: {elapsed:.3f}s ({steps} steps), n_cells={n}")

    if args.time_average and t_acc > 0:
        rho = (rho_acc / t_acc).astype(np.float32)
        speed = (speed_acc / t_acc).astype(np.float32)
        flow = (flow_acc / t_acc).astype(np.float32)
        log(f"시간평균 적용: T={t_acc:.1f}s")

    # 출력: edge 집계 CSV (SUMO 비교용)
    eout = Path(args.edge_output_csv)
    eout.parent.mkdir(parents=True, exist_ok=True)
    agg = aggregate_cells_to_edges(cnet, rho, speed, flow)
    with eout.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["edge_id", "lanes", "density_veh_per_m", "speed_mps", "flow_veh_per_s"])
        for eid, (dens, msp, fl, lanes) in agg.items():
            w.writerow([eid, lanes, dens, msp, fl])
    log(f"edge 집계 저장: {eout}")

    # 출력: per-cell CSV (선택)
    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["cell_id", "lane_id", "edge_id", "cell_local", "length_m",
                        "vmax_mps", "density_veh_per_m", "speed_mps", "flow_veh_per_s"])
            for i in range(n):
                li = int(cnet.cell_lane[i])
                ei = int(lane_net.lane_edge[li])
                w.writerow([
                    i, lane_net.lane_ids[li], lane_net.edge_ids[ei], int(cnet.cell_local[i]),
                    float(cnet.length[i]), float(cnet.vmax[i]),
                    float(rho[i]), float(speed[i]), float(flow[i]),
                ])
        log(f"cell 결과 저장: {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-cell CTM CPU traffic simulator (mesoscopic-leaning)")
    p.add_argument("--net-file", default="./map_import/gangnam4_generated.net.xml")
    p.add_argument("--route-file", default="./map_import/gangnam4_generated.gpu_compatible.rou.xml")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--sim-time", type=float, default=0.0)
    p.add_argument("--sim-duration", type=float, default=86400.0)
    p.add_argument("--cell-length", type=float, default=15.0, help="목표 cell 길이(m). 차선당 cell 수 = round(lane.length/이 값)")
    p.add_argument("--jam-density-per-lane", type=float, default=0.18)
    p.add_argument("--init-density", type=float, default=0.03)
    p.add_argument("--source-demand", type=float, default=0.02)
    p.add_argument("--vmax-scale", type=float, default=1.0)
    p.add_argument("--major-left-factor", type=float, default=1.0)
    p.add_argument("--minor-factor", type=float, default=1.0)
    p.add_argument("--time-average", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=200)
    p.add_argument("--output-csv", default="")
    p.add_argument("--edge-output-csv", default="./gangnam4_cuda/results/cell_state_cpu.edge.csv")
    args = p.parse_args()
    run_sim(args)


if __name__ == "__main__":
    main()
