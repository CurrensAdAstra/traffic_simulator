#!/usr/bin/env python3
"""
Multi-cell CTM 공용 모듈 — 차선(lane)을 다시 공간 cell들로 세분화한 메소스코픽-지향 표현.

기존 lane_common의 LaneNet 위에 한 차선당 N(≥1)개의 cell을 두고,
- intra-lane(같은 lane의 cell[i] → cell[i+1])
- inter-lane(상류 lane의 마지막 cell → 하류 lane의 첫 cell, lane_net.connection에서 상속)
의 두 종류 연결을 갖는다. CTM 갱신은 cell 단위로 수행하며, 연결 단위(priority/dir/junction)는
inter-lane 측에서만 그대로 따른다. intra-lane은 항상 major-straight로 취급(자유로운 종방향 흐름).

장점:
 - 한 차선 안에서 head/tail 밀도 분리 → 좌회전 차선 앞쪽 큐 누적, 뒷쪽은 여전히 흘러가는 현상 가능
 - 차선 진출입 지점 spillback이 cell-by-cell로 부드럽게 전파됨
한계:
 - 여전히 결정론적 LWR/CTM. stop-and-go 같은 시간적 진동은 표현 못 함 → 진짜 mesoscopic(packet)이 아님
 - 측면 차선변경은 일단 cell 단위가 아닌 lane 단위로(또는 생략) 처리
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lane_common import LaneNet, load_lane_net, log


@dataclass
class CellNet:
    # 기반 LaneNet 보관 — edge/lane id, 집계 등에 필요
    lane_net: LaneNet
    # cell 식별 / 속성
    n_cells: int
    cell_lane: np.ndarray         # [n_cells] int32, cell이 속한 lane
    cell_local: np.ndarray        # [n_cells] int32, lane 안에서의 위치(0=head)
    length: np.ndarray            # [n_cells] float32
    vmax: np.ndarray              # [n_cells] float32
    # lane → cells 매핑
    lane_cell_first: np.ndarray   # [n_lanes] int32
    lane_cell_count: np.ndarray   # [n_lanes] int32
    # cell-level 연결 (intra + inter)
    n_conn: int
    conn_src: np.ndarray          # [n_conn] int32 cell index
    conn_dst: np.ndarray          # [n_conn] int32 cell index
    conn_split: np.ndarray        # [n_conn] float32 (intra=1.0, inter=lane conn_split)
    conn_priority: np.ndarray     # [n_conn] int8 (intra=1 major, inter=lane conn_priority)
    conn_dir: np.ndarray          # [n_conn] int8 (intra=DIR_S, inter=lane conn_dir)
    is_intra: np.ndarray          # [n_conn] int8 (1=intra-lane, 0=inter-lane) — HCM penalty에서 intra는 제외
    # gather용 CSR
    in_conn_ptr: np.ndarray       # [n_cells+1]
    in_conn_idx: np.ndarray       # [n_conn]
    out_conn_ptr: np.ndarray      # [n_cells+1]
    out_conn_idx: np.ndarray      # [n_conn]
    # 첫 cell(=lane의 head)만 외부에서 source_demand를 받음
    no_incoming_mask: np.ndarray  # [n_cells] int8/bool


def build_cell_net(lane_net: LaneNet, target_cell_length: float = 15.0) -> CellNet:
    """LaneNet을 받아 target_cell_length(m)에 맞춰 cell들을 만든다.

    각 lane의 cell 수: max(1, round(lane.length / target_cell_length)).
    Cell length = lane.length / cell 수. Cell의 vmax는 lane의 vmax를 그대로.
    """
    n_lanes = lane_net.n_lanes
    lane_lengths = lane_net.length_m
    # 차선당 cell 수
    lane_cell_count = np.maximum(
        1, np.round(lane_lengths / float(target_cell_length))
    ).astype(np.int32)
    # 차선별 첫 cell index
    lane_cell_first = np.concatenate(
        [[0], np.cumsum(lane_cell_count)[:-1]]
    ).astype(np.int32)
    n_cells = int(lane_cell_count.sum())

    # cell 속성
    cell_lane = np.zeros(n_cells, dtype=np.int32)
    cell_local = np.zeros(n_cells, dtype=np.int32)
    length = np.zeros(n_cells, dtype=np.float32)
    vmax = np.zeros(n_cells, dtype=np.float32)
    for li in range(n_lanes):
        first = int(lane_cell_first[li])
        count = int(lane_cell_count[li])
        cl = float(lane_lengths[li]) / count
        vm = float(lane_net.vmax_mps[li])
        for ci in range(count):
            idx = first + ci
            cell_lane[idx] = li
            cell_local[idx] = ci
            length[idx] = cl
            vmax[idx] = vm

    # --- 연결 구축 ---
    # 1) intra-lane: 각 lane의 cell[i] → cell[i+1]
    intra_src_list: list[int] = []
    intra_dst_list: list[int] = []
    for li in range(n_lanes):
        first = int(lane_cell_first[li])
        count = int(lane_cell_count[li])
        for ci in range(count - 1):
            intra_src_list.append(first + ci)
            intra_dst_list.append(first + ci + 1)
    intra_n = len(intra_src_list)

    # 2) inter-lane: lane_net의 lane→lane 연결을 (last_cell(src), first_cell(dst)) 로 매핑
    lane_last_cell = (lane_cell_first + lane_cell_count - 1).astype(np.int32)
    inter_src = lane_last_cell[lane_net.conn_src]
    inter_dst = lane_cell_first[lane_net.conn_dst]
    inter_n = lane_net.n_conn

    # 3) 합치기
    n_conn = intra_n + inter_n
    conn_src = np.concatenate(
        [np.asarray(intra_src_list, dtype=np.int32), inter_src.astype(np.int32)]
    )
    conn_dst = np.concatenate(
        [np.asarray(intra_dst_list, dtype=np.int32), inter_dst.astype(np.int32)]
    )
    conn_split = np.concatenate([
        np.ones(intra_n, dtype=np.float32),         # intra: 전부 다음 cell로
        lane_net.conn_split.astype(np.float32),     # inter: lane 단위 split 그대로
    ])
    conn_priority = np.concatenate([
        np.ones(intra_n, dtype=np.int8),            # intra: 항상 major
        lane_net.conn_priority.astype(np.int8),
    ])
    # lane_common.DIR_S = 1 (straight)
    from lane_common import DIR_S
    conn_dir = np.concatenate([
        np.full(intra_n, DIR_S, dtype=np.int8),
        lane_net.conn_dir.astype(np.int8),
    ])
    is_intra = np.concatenate([
        np.ones(intra_n, dtype=np.int8),
        np.zeros(inter_n, dtype=np.int8),
    ])

    # 4) CSR
    in_count = np.bincount(conn_dst, minlength=n_cells)
    out_count = np.bincount(conn_src, minlength=n_cells)
    in_conn_ptr = np.concatenate([[0], np.cumsum(in_count)]).astype(np.int32)
    out_conn_ptr = np.concatenate([[0], np.cumsum(out_count)]).astype(np.int32)
    in_conn_idx = np.argsort(conn_dst, kind="stable").astype(np.int32)
    out_conn_idx = np.argsort(conn_src, kind="stable").astype(np.int32)

    # 5) no_incoming_mask: 차선 head이면서 lane_net에서 해당 차선이 no_incoming일 때만 외부 inflow
    # lane-level no_incoming
    lane_in_count = np.diff(lane_net.in_ptr)
    lane_no_in = lane_in_count == 0
    no_incoming_mask = np.zeros(n_cells, dtype=np.int8)
    head_cells = lane_cell_first  # 각 lane의 첫 cell
    # head_cells에서, lane이 no_incoming인 cell만 마킹
    no_incoming_mask[head_cells[lane_no_in]] = 1

    log(
        f"cell net 빌드: lanes={n_lanes}, cells={n_cells} (target_cell_length={target_cell_length}m), "
        f"intra={intra_n}, inter={inter_n}, total_conn={n_conn}, "
        f"head-cells with source={int(no_incoming_mask.sum())}"
    )

    return CellNet(
        lane_net=lane_net,
        n_cells=n_cells,
        cell_lane=cell_lane,
        cell_local=cell_local,
        length=length,
        vmax=vmax,
        lane_cell_first=lane_cell_first,
        lane_cell_count=lane_cell_count,
        n_conn=n_conn,
        conn_src=conn_src,
        conn_dst=conn_dst,
        conn_split=conn_split,
        conn_priority=conn_priority,
        conn_dir=conn_dir,
        is_intra=is_intra,
        in_conn_ptr=in_conn_ptr,
        in_conn_idx=in_conn_idx,
        out_conn_ptr=out_conn_ptr,
        out_conn_idx=out_conn_idx,
        no_incoming_mask=no_incoming_mask,
    )


def cells_to_lanes(cell_net: CellNet, rho_c: np.ndarray, speed_c: np.ndarray, flow_c: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """cell 상태를 lane 단위로 길이-가중 평균하여 집계."""
    L = cell_net.lane_net.n_lanes
    rho_lane = np.zeros(L, dtype=np.float32)
    speed_lane = np.zeros(L, dtype=np.float32)
    flow_lane = np.zeros(L, dtype=np.float32)
    for li in range(L):
        s = int(cell_net.lane_cell_first[li])
        e = s + int(cell_net.lane_cell_count[li])
        cl = cell_net.length[s:e]
        wsum = float(cl.sum()) or 1.0
        rho_lane[li] = float((rho_c[s:e] * cl).sum() / wsum)
        speed_lane[li] = float((speed_c[s:e] * cl).sum() / wsum)
        flow_lane[li] = float((flow_c[s:e] * cl).sum() / wsum)
    return rho_lane, speed_lane, flow_lane


def aggregate_cells_to_edges(cell_net: CellNet, rho_c: np.ndarray, speed_c: np.ndarray, flow_c: np.ndarray):
    """cell → edge 직접 집계 (SUMO 비교용)."""
    rho_lane, speed_lane, flow_lane = cells_to_lanes(cell_net, rho_c, speed_c, flow_c)
    from lane_common import aggregate_to_edges
    return aggregate_to_edges(cell_net.lane_net, rho_lane, speed_lane, flow_lane)
