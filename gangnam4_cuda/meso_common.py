#!/usr/bin/env python3
"""
Mesoscopic(차량 단위) 시뮬레이터 공용 로더.

CTM 계열(밀도 연속체)과 달리 개별 차량(vehicle)을 추적한다.
 - 각 차량은 route(edge 시퀀스)를 따라 edge에서 edge로 이동
 - edge는 spatial-queue 모델: running 구간 + 하류 끝 queue
 - edge 통과시간은 밀도 기반 속도로 결정(메소스코픽 speed-density)
 - 하류 전이는 (송신 saturation capacity) + (수신 storage space)로 제약

이 표현의 핵심 이점: SUMO tripinfo와 직접 비교 가능한 **차량별 통행시간**을 산출.
edge 단위 평균(밀도/속도/유량)도 부산물로 나오므로 기존 edgedata 비교도 유지.
"""

from __future__ import annotations

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
class MesoNet:
    # edge 식별/속성
    edge_ids: list[str]
    edge_to_idx: dict[str, int]
    n_edges: int
    length_m: np.ndarray        # [E] float32
    lanes: np.ndarray           # [E] float32
    vmax_mps: np.ndarray        # [E] float32
    jam_storage: np.ndarray     # [E] float32, edge가 담을 수 있는 최대 차량 수
    sat_cap_per_s: np.ndarray   # [E] float32, edge 유출 saturation flow (veh/s)
    # 차량/route (flat)
    n_veh: int
    veh_depart: np.ndarray      # [V] float32, 출발시각(정렬됨)
    veh_route_off: np.ndarray   # [V] int32, route_edges 시작 offset
    veh_route_len: np.ndarray   # [V] int32, route 길이(edge 수)
    veh_ids: list[str]          # [V] 원본 id (정렬 순서)
    route_edges: np.ndarray     # [sum(len)] int32, 모든 차량 route를 이어붙인 edge 인덱스


def _parse_edges(root) -> tuple[list[str], dict, np.ndarray, np.ndarray, np.ndarray]:
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
    edge_to_idx = {eid: i for i, eid in enumerate(edge_ids)}
    return (edge_ids, edge_to_idx,
            np.asarray(length, dtype=np.float32),
            np.asarray(lanes, dtype=np.float32),
            np.asarray(vmax, dtype=np.float32))


def load_meso_net(
    net_file: Path,
    route_file: Path,
    jam_density_per_lane: float = 0.18,
    sat_flow_per_lane: float = 0.5,
    vmax_scale: float = 1.0,
    max_vehicles: int = 0,
) -> MesoNet:
    if not net_file.exists():
        fail(f"net 파일 없음: {net_file}")
    if not route_file.exists():
        fail(f"route 파일 없음: {route_file}")

    root = ET.parse(net_file).getroot()
    edge_ids, edge_to_idx, length, lanes, vmax = _parse_edges(root)
    if vmax_scale != 1.0:
        vmax = (vmax * np.float32(vmax_scale)).astype(np.float32)
    n_edges = len(edge_ids)

    jam_storage = (length * lanes * np.float32(jam_density_per_lane)).astype(np.float32)
    jam_storage = np.maximum(jam_storage, 1.0).astype(np.float32)
    sat_cap_per_s = (lanes * np.float32(sat_flow_per_lane)).astype(np.float32)

    # --- route id → edge 인덱스 시퀀스 ---
    route_seq: dict[str, list[int]] = {}
    for _, el in ET.iterparse(route_file, events=("end",)):
        if el.tag == "route":
            rid = el.get("id", "")
            if rid:
                seq = [edge_to_idx[e] for e in el.get("edges", "").split() if e in edge_to_idx]
                if seq:
                    route_seq[rid] = seq
            el.clear()
        elif el.tag == "vehicle":
            # 일부 net은 route를 vehicle 내부에 둠 — 그 경우도 처리(아래 vehicle pass에서)
            pass

    # --- 차량 목록 ---
    veh_depart_l: list[float] = []
    veh_route_l: list[list[int]] = []
    veh_ids_l: list[str] = []
    for _, el in ET.iterparse(route_file, events=("end",)):
        if el.tag == "vehicle":
            rid = el.get("route", "")
            seq = route_seq.get(rid)
            if seq is None:
                # vehicle 내부 route?
                rn = el.find("route")
                if rn is not None:
                    seq = [edge_to_idx[e] for e in rn.get("edges", "").split() if e in edge_to_idx]
            if seq:
                veh_depart_l.append(float(el.get("depart", "0")))
                veh_route_l.append(seq)
                veh_ids_l.append(el.get("id", ""))
            el.clear()

    if max_vehicles and len(veh_depart_l) > max_vehicles:
        # 출발 순으로 앞쪽 max_vehicles만(테스트용)
        order = np.argsort(veh_depart_l)[:max_vehicles]
        veh_depart_l = [veh_depart_l[i] for i in order]
        veh_route_l = [veh_route_l[i] for i in order]
        veh_ids_l = [veh_ids_l[i] for i in order]

    # 출발시각 정렬
    order = np.argsort(np.asarray(veh_depart_l, dtype=np.float64), kind="stable")
    veh_depart = np.asarray(veh_depart_l, dtype=np.float32)[order]
    veh_ids = [veh_ids_l[i] for i in order]
    veh_route_sorted = [veh_route_l[i] for i in order]

    # flat route 배열
    n_veh = len(veh_route_sorted)
    veh_route_len = np.asarray([len(r) for r in veh_route_sorted], dtype=np.int32)
    veh_route_off = np.concatenate([[0], np.cumsum(veh_route_len)[:-1]]).astype(np.int32) if n_veh else np.zeros(0, dtype=np.int32)
    route_edges = np.concatenate([np.asarray(r, dtype=np.int32) for r in veh_route_sorted]) if n_veh else np.zeros(0, dtype=np.int32)

    log(f"meso net 로드: edges={n_edges}, vehicles={n_veh}, "
        f"route_edges_total={len(route_edges)}, jam_density/lane={jam_density_per_lane}, "
        f"sat_flow/lane={sat_flow_per_lane} veh/s, vmax_scale={vmax_scale}")

    return MesoNet(
        edge_ids=edge_ids,
        edge_to_idx=edge_to_idx,
        n_edges=n_edges,
        length_m=length,
        lanes=lanes,
        vmax_mps=vmax,
        jam_storage=jam_storage,
        sat_cap_per_s=sat_cap_per_s,
        n_veh=n_veh,
        veh_depart=veh_depart,
        veh_route_off=veh_route_off,
        veh_route_len=veh_route_len,
        veh_ids=veh_ids,
        route_edges=route_edges,
    )
