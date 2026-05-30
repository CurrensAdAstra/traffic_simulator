#!/usr/bin/env python3
"""
차선(lane) 단위 시뮬레이션 공용 모듈.

edge 단위 엔진(`cuda_edge_simulator.py` / `cpu_edge_simulator_mt.py`)의
`load_net` / `build_source_demand_by_edge` 를 lane 단위로 확장한 것.

- 모든 lane을 전역 배열(크기 L = Σ lanes)로 평탄화
- lane→lane 종방향(longitudinal) 연결을 connection의 fromLane/toLane에서 직접 구성
- connection의 dir(l/s/r/t)로부터 lane별 회전(turn) 가능 방향을 추출
- route 파일에서 edge별 회전 수요를 집계 → lane별 목표 밀도 분배(target_share)

CPU/GPU 엔진이 이 모듈을 공유하여 모델/입력 해석을 일치시킨다.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def log(msg: str) -> None:
    print(f"[LOG] {msg}")


def fail(msg: str) -> None:
    print(f"[ERROR] {msg}")
    raise SystemExit(1)


# 회전 방향 비트: L(좌), S(직진), R(우), T(유턴)
DIR_L, DIR_S, DIR_R, DIR_T = 0, 1, 2, 3
_DIR_MAP = {"l": DIR_L, "L": DIR_L, "s": DIR_S, "r": DIR_R, "R": DIR_R, "t": DIR_T}


def dir_to_bit(d: str) -> int:
    """SUMO connection dir 문자 → 방향 인덱스(미지원이면 -1)."""
    return _DIR_MAP.get(d, -1)


@dataclass
class LaneNet:
    # edge 식별
    edge_ids: list[str]
    edge_to_idx: dict[str, int]
    # lane 식별 / 속성 (전역 lane 인덱스 0..L-1)
    lane_ids: list[str]
    n_edges: int
    n_lanes: int
    lane_edge: np.ndarray      # [L] int32, 소속 edge 인덱스
    lane_local: np.ndarray     # [L] int32, edge 내 lane index
    length_m: np.ndarray       # [L] float32
    vmax_mps: np.ndarray       # [L] float32
    caps: np.ndarray           # [L] int32, 회전 가능 방향 비트마스크
    # 종방향 incoming CSR (lane→lane)
    in_ptr: np.ndarray         # [L+1] int32
    in_lanes: np.ndarray       # [nnz] int32
    in_w: np.ndarray           # [nnz] float32
    # 횡방향(차선변경) 인접 CSR (같은 edge의 local±1 lane)
    lat_ptr: np.ndarray        # [L+1] int32
    lat_neighbors: np.ndarray  # [nnz2] int32
    # edge→lane 매핑 (집계/검증용)
    edge_lane_ptr: np.ndarray  # [n_edges+1] int32
    edge_lanes: np.ndarray     # [L] int32, edge별 lane 전역 인덱스(정렬)
    # 연결(connection) 평탄 배열 — CTM(sending/receiving) 갱신에 사용
    n_conn: int
    conn_src: np.ndarray       # [n_conn] int32, 송신 lane
    conn_dst: np.ndarray       # [n_conn] int32, 수신 lane
    conn_split: np.ndarray     # [n_conn] float32, 송신 측 분배 비율(1/outdeg)


def load_lane_net(net_file: Path) -> LaneNet:
    if not net_file.exists():
        fail(f"net 파일 없음: {net_file}")

    root = ET.parse(net_file).getroot()

    edge_ids: list[str] = []
    # 임시 lane 누적
    lane_ids: list[str] = []
    lane_edge_l: list[int] = []
    lane_local_l: list[int] = []
    length_l: list[float] = []
    vmax_l: list[float] = []
    # sumo lane id("{edge}_{idx}") → 전역 lane 인덱스
    lane_uid: dict[str, int] = {}
    # edge 인덱스 → 소속 전역 lane 인덱스 목록
    edge_lane_list: list[list[int]] = []

    edge_to_idx: dict[str, int] = {}

    for e in root.findall("edge"):
        eid = e.get("id", "")
        if not eid or eid.startswith(":"):
            continue
        if e.get("function", "") in {"internal", "crossing", "walkingarea"}:
            continue

        lane_nodes = e.findall("lane")
        if not lane_nodes:
            continue

        ei = len(edge_ids)
        edge_ids.append(eid)
        edge_to_idx[eid] = ei
        my_lanes: list[int] = []

        for ln in lane_nodes:
            local = int(ln.get("index", str(len(my_lanes))))
            lid = ln.get("id", f"{eid}_{local}")
            gidx = len(lane_ids)
            lane_ids.append(lid)
            lane_uid[lid] = gidx
            lane_edge_l.append(ei)
            lane_local_l.append(local)
            length_l.append(max(float(ln.get("length", "1")), 1.0))
            vmax_l.append(max(float(ln.get("speed", "13.9")), 0.1))
            my_lanes.append(gidx)

        edge_lane_list.append(my_lanes)

    n_edges = len(edge_ids)
    n_lanes = len(lane_ids)
    if n_lanes == 0:
        fail("유효 lane이 0개")

    lane_edge = np.asarray(lane_edge_l, dtype=np.int32)
    lane_local = np.asarray(lane_local_l, dtype=np.int32)
    length_m = np.asarray(length_l, dtype=np.float32)
    vmax_mps = np.asarray(vmax_l, dtype=np.float32)

    # ---- 종방향 lane→lane 연결 + lane별 회전 가능 방향 ----
    caps = np.zeros(n_lanes, dtype=np.int32)
    out_count = np.zeros(n_lanes, dtype=np.int32)
    raw_conns: list[tuple[int, int]] = []  # (src_lane, dst_lane)

    for c in root.findall("connection"):
        f = c.get("from", "")
        t = c.get("to", "")
        if f not in edge_to_idx or t not in edge_to_idx:
            continue
        fl = c.get("fromLane")
        tl = c.get("toLane")
        if fl is None or tl is None:
            continue
        src = lane_uid.get(f"{f}_{fl}")
        dst = lane_uid.get(f"{t}_{tl}")
        if src is None or dst is None:
            continue
        raw_conns.append((src, dst))
        out_count[src] += 1
        b = dir_to_bit(c.get("dir", ""))
        if b >= 0:
            caps[src] |= (1 << b)

    # 연결 평탄 배열(원본 순서 유지) — CTM 갱신용
    conn_src_list = [s for s, _ in raw_conns]
    conn_dst_list = [d for _, d in raw_conns]
    conn_split_list = [1.0 / max(int(out_count[s]), 1) for s, _ in raw_conns]

    incoming: list[list[tuple[int, float]]] = [[] for _ in range(n_lanes)]
    for src, dst in raw_conns:
        w = 1.0 / max(int(out_count[src]), 1)
        incoming[dst].append((src, float(w)))

    in_ptr = [0]
    in_lanes: list[int] = []
    in_w: list[float] = []
    for arr in incoming:
        for src, w in arr:
            in_lanes.append(src)
            in_w.append(w)
        in_ptr.append(len(in_lanes))

    # ---- 횡방향 인접(같은 edge, local±1) ----
    # local index → 전역 lane 매핑(edge별)
    lat_ptr = [0]
    lat_neighbors: list[int] = []
    edge_lane_ptr = [0]
    edge_lanes_flat: list[int] = []

    # local→global 조회를 위해 edge별 dict 구성
    for ei, lanes in enumerate(edge_lane_list):
        local_to_g = {int(lane_local[g]): g for g in lanes}
        # edge_lanes는 local 순으로 정렬
        ordered = [local_to_g[k] for k in sorted(local_to_g.keys())]
        edge_lanes_flat.extend(ordered)
        edge_lane_ptr.append(len(edge_lanes_flat))

    # lat CSR은 전역 lane 순서(0..L-1)대로 작성해야 하므로 별도 루프
    # 각 lane의 인접 lane(local±1)을 같은 edge에서 찾는다.
    local_lookup: list[dict[int, int]] = []
    for ei, lanes in enumerate(edge_lane_list):
        local_lookup.append({int(lane_local[g]): g for g in lanes})

    for g in range(n_lanes):
        ei = int(lane_edge[g])
        loc = int(lane_local[g])
        lk = local_lookup[ei]
        for nb_loc in (loc - 1, loc + 1):
            nb = lk.get(nb_loc)
            if nb is not None:
                lat_neighbors.append(nb)
        lat_ptr.append(len(lat_neighbors))

    log(
        f"lane net 로드 완료: edges={n_edges}, lanes={n_lanes}, "
        f"connections={len(raw_conns)}, lat_edges={len(lat_neighbors)}"
    )

    return LaneNet(
        edge_ids=edge_ids,
        edge_to_idx=edge_to_idx,
        lane_ids=lane_ids,
        n_edges=n_edges,
        n_lanes=n_lanes,
        lane_edge=lane_edge,
        lane_local=lane_local,
        length_m=length_m,
        vmax_mps=vmax_mps,
        caps=caps,
        in_ptr=np.asarray(in_ptr, dtype=np.int32),
        in_lanes=np.asarray(in_lanes, dtype=np.int32),
        in_w=np.asarray(in_w, dtype=np.float32),
        lat_ptr=np.asarray(lat_ptr, dtype=np.int32),
        lat_neighbors=np.asarray(lat_neighbors, dtype=np.int32),
        edge_lane_ptr=np.asarray(edge_lane_ptr, dtype=np.int32),
        edge_lanes=np.asarray(edge_lanes_flat, dtype=np.int32),
        n_conn=len(raw_conns),
        conn_src=np.asarray(conn_src_list, dtype=np.int32),
        conn_dst=np.asarray(conn_dst_list, dtype=np.int32),
        conn_split=np.asarray(conn_split_list, dtype=np.float32),
    )


def _build_edge_pair_dir(net_file: Path, edge_to_idx: dict[str, int]) -> dict[tuple[int, int], int]:
    """(from_edge_idx, to_edge_idx) → 방향 비트. 같은 쌍에 여러 connection이 있으면 첫 dir 사용."""
    root = ET.parse(net_file).getroot()
    pair_dir: dict[tuple[int, int], int] = {}
    for c in root.findall("connection"):
        f = c.get("from", "")
        t = c.get("to", "")
        fi = edge_to_idx.get(f)
        ti = edge_to_idx.get(t)
        if fi is None or ti is None:
            continue
        b = dir_to_bit(c.get("dir", ""))
        if b < 0:
            continue
        pair_dir.setdefault((fi, ti), b)
    return pair_dir


def build_demand_and_target(
    net: LaneNet,
    net_file: Path,
    route_file: Path | None,
    sim_duration: float,
    source_demand_default: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    route 파일로부터:
      - source_demand[L]: 진입 차선별 inflow 비율(veh/s). route의 첫 edge에 진입.
      - target_share[L]: edge 내 lane별 목표 밀도 비중(회전 수요 기반, edge별 합=1).
    route가 없으면 source_demand는 상수 fallback, target_share는 균등 분배.
    반환: (source_demand, target_share, vehicle_count)
    """
    L = net.n_lanes
    E = net.n_edges

    # 기본값(균등 target, 상수 source)
    target_share = np.zeros(L, dtype=np.float32)
    for ei in range(E):
        s = int(net.edge_lane_ptr[ei])
        e = int(net.edge_lane_ptr[ei + 1])
        k = e - s
        if k > 0:
            target_share[net.edge_lanes[s:e]] = 1.0 / k

    source_demand = np.full(L, float(source_demand_default), dtype=np.float32)

    if route_file is None or not Path(route_file).exists():
        log("route_file 없음: 균등 target_share + 상수 source_demand 사용")
        return source_demand, target_share, 0

    route_file = Path(route_file)
    pair_dir = _build_edge_pair_dir(net_file, net.edge_to_idx)

    # pass1: route id별 차량 수
    veh_count: Counter[str] = Counter()
    vehicle_count = 0
    for _, el in ET.iterparse(route_file, events=("end",)):
        if el.tag == "vehicle":
            vehicle_count += 1
            rid = el.get("route")
            if rid:
                veh_count[rid] += 1
            el.clear()

    # pass2: route별 edge열 → 첫 edge 수요 + 회전 수요 집계(차량 수 가중)
    edge_first_demand = np.zeros(E, dtype=np.float64)         # edge별 진입 차량 수
    turn_dem = np.zeros((E, 4), dtype=np.float64)             # edge별 방향(L/S/R/T) 수요
    for _, el in ET.iterparse(route_file, events=("end",)):
        if el.tag != "route":
            el.clear()
            continue
        rid = el.get("id", "")
        w = veh_count.get(rid, 0)
        if w <= 0:
            el.clear()
            continue
        edges = el.get("edges", "").split()
        if edges:
            fi = net.edge_to_idx.get(edges[0])
            if fi is not None:
                edge_first_demand[fi] += w
            for a, b in zip(edges, edges[1:]):
                ia = net.edge_to_idx.get(a)
                ib = net.edge_to_idx.get(b)
                if ia is None or ib is None:
                    continue
                d = pair_dir.get((ia, ib))
                if d is not None:
                    turn_dem[ia, d] += w
        el.clear()

    # source_demand[L]: edge 진입 수요를 lane에 균등 분배 후 sim_duration으로 환산
    dur = sim_duration if sim_duration > 0 else 1.0
    source_demand = np.zeros(L, dtype=np.float32)
    for ei in range(E):
        s = int(net.edge_lane_ptr[ei])
        e = int(net.edge_lane_ptr[ei + 1])
        k = e - s
        if k > 0 and edge_first_demand[ei] > 0:
            per_lane = (edge_first_demand[ei] / dur) / k
            source_demand[net.edge_lanes[s:e]] = per_lane

    # target_share[L]: edge별 방향 수요를 capable lane에 분배
    for ei in range(E):
        s = int(net.edge_lane_ptr[ei])
        e = int(net.edge_lane_ptr[ei + 1])
        lanes = net.edge_lanes[s:e]
        k = len(lanes)
        if k == 0:
            continue
        dem = turn_dem[ei]
        total = float(dem.sum())
        if total <= 0:
            target_share[lanes] = 1.0 / k  # 수요 없음 → 균등
            continue
        acc = np.zeros(k, dtype=np.float64)
        for d in range(4):
            if dem[d] <= 0:
                continue
            frac = dem[d] / total
            capable = [j for j, g in enumerate(lanes) if (int(net.caps[g]) >> d) & 1]
            if capable:
                share = frac / len(capable)
                for j in capable:
                    acc[j] += share
            else:
                # 해당 방향을 가진 lane이 없으면 전체에 균등 분배(fallback)
                acc += frac / k
        ssum = acc.sum()
        if ssum > 0:
            acc /= ssum
        else:
            acc[:] = 1.0 / k
        target_share[lanes] = acc.astype(np.float32)

    log(f"route 반영: vehicles={vehicle_count}, routes={len(veh_count)}, sim_duration={sim_duration}")
    return source_demand, target_share, vehicle_count


def aggregate_to_edges(
    net: LaneNet, rho: np.ndarray, speed: np.ndarray, flow: np.ndarray
) -> dict[str, tuple[float, float, float, float]]:
    """
    lane 결과를 edge 단위로 집계.
    edge별: (총 밀도 합, lane-length 가중 평균 속도, 유량 합, lane 수)
    반환: edge_id → (density_sum, mean_speed, flow_sum, lanes)
    """
    out: dict[str, tuple[float, float, float, float]] = {}
    for ei, eid in enumerate(net.edge_ids):
        s = int(net.edge_lane_ptr[ei])
        e = int(net.edge_lane_ptr[ei + 1])
        lanes = net.edge_lanes[s:e]
        if len(lanes) == 0:
            continue
        w = net.length_m[lanes]
        wsum = float(w.sum()) or 1.0
        mean_speed = float((speed[lanes] * w).sum() / wsum)
        out[eid] = (
            float(rho[lanes].sum()),
            mean_speed,
            float(flow[lanes].sum()),
            float(len(lanes)),
        )
    return out
