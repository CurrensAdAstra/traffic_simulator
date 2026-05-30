#!/usr/bin/env python3
"""
SUMO vs CPU-MT vs GPU(edge-병렬) 동일 조건 실행/비교 래퍼.

동일 조건 정의:
- 동일 net-file
- 동일 route-file(vehicles)
- 동일 시뮬레이션 기간(초)
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict


def log(msg: str) -> None:
    print(f"[LOG] {msg}")


def fail(msg: str) -> None:
    print(f"[ERROR] {msg}")
    raise SystemExit(1)


def run_cmd(cmd: list[str]) -> float:
    t0 = time.perf_counter()
    log("CMD: " + " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    dt = time.perf_counter() - t0
    if rc != 0:
        fail(f"명령 실패(rc={rc}): {' '.join(cmd)}")
    return dt


def to_cfg_relative(path: Path, cfg_path: Path) -> str:
    """SUMO cfg 기준 상대경로로 변환."""
    cfg_dir = cfg_path.parent.resolve()
    p = path.resolve()
    return os.path.relpath(str(p), str(cfg_dir))


def count_net_stats(net_file: Path) -> tuple[int, int, int]:
    root = ET.parse(net_file).getroot()
    edges_all = root.findall("edge")
    conns_all = root.findall("connection")

    ext = 0
    for e in edges_all:
        eid = e.get("id", "")
        if not eid or eid.startswith(":"):
            continue
        if e.get("function", "") in {"internal", "crossing", "walkingarea"}:
            continue
        if not e.findall("lane"):
            continue
        ext += 1

    return len(edges_all), len(conns_all), ext


def count_vehicles(route_file: Path) -> int:
    n = 0
    for _, elem in ET.iterparse(route_file, events=("end",)):
        if elem.tag == "vehicle":
            n += 1
            elem.clear()
    return n


def parse_sumo_tripinfo(tripinfo_file: Path) -> tuple[int, float, float]:
    n = 0
    duration_sum = 0.0
    speed_sum = 0.0
    for _, elem in ET.iterparse(tripinfo_file, events=("end",)):
        if elem.tag == "tripinfo":
            n += 1
            duration = float(elem.attrib.get("duration", 0.0))
            route_len = float(elem.attrib.get("routeLength", 0.0))
            duration_sum += duration
            speed_sum += (route_len / duration) if duration > 0 else 0.0
            elem.clear()

    mean_duration = duration_sum / n if n else 0.0
    mean_speed = speed_sum / n if n else 0.0
    return n, mean_duration, mean_speed


def parse_edge_csv(edge_csv: Path) -> tuple[int, float, float]:
    n = 0
    speed_sum = 0.0
    tt_sum = 0.0
    with edge_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            n += 1
            speed_sum += float(row["speed_mps"])
            tt_sum += float(row["travel_time_s"])
    return n, (speed_sum / n if n else 0.0), (tt_sum / n if n else 0.0)


def read_edge_metric_map(edge_csv: Path) -> Dict[str, dict[str, float]]:
    out: Dict[str, dict[str, float]] = {}
    with edge_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            eid = row["edge_id"]
            out[eid] = {
                "speed_mps": float(row["speed_mps"]),
                "travel_time_s": float(row["travel_time_s"]),
                "density_veh_per_m": float(row["density_veh_per_m"]),
                "flow_veh_per_s": float(row["flow_veh_per_s"]),
            }
    return out


def parse_sumo_edgedata_last_interval(edgedata_file: Path) -> Dict[str, dict[str, float]]:
    """SUMO edgedata xml에서 마지막 interval의 edge 지표를 읽는다.

    단위 정규화: SUMO edgeData는 density를 veh/km, flow를 veh/h로 보고하므로
    엔진 출력(veh/m, veh/s)과 일치시키기 위해 여기서 환산한다.
      - density: veh/km  → veh/m  (÷1000)
      - flow:    veh/h   → veh/s  (÷3600)
      - speed:   m/s     (그대로)
    """
    tree = ET.parse(edgedata_file)
    root = tree.getroot()
    intervals = root.findall("interval")
    if not intervals:
        return {}

    last = intervals[-1]
    out: Dict[str, dict[str, float]] = {}
    for e in last.findall("edge"):
        eid = e.get("id", "")
        if not eid:
            continue
        density_per_km = float(e.get("density", "0") or 0.0)
        flow_per_h = float(e.get("flow", "0") or 0.0)
        out[eid] = {
            "speed_mps": float(e.get("speed", "0") or 0.0),
            "travel_time_s": float(e.get("traveltime", "0") or 0.0),
            "density_veh_per_m": density_per_km / 1000.0,
            "flow_veh_per_s": flow_per_h / 3600.0,
            "sampled_seconds": float(e.get("sampledSeconds", "0") or 0.0),
        }
    return out


def write_edgewise_report(
    path: Path,
    sumo_edge: Dict[str, dict[str, float]],
    cpu_edge: Dict[str, dict[str, float]],
    gpu_edge: Dict[str, dict[str, float]],
) -> None:
    fieldnames = [
        "edge_id",
        "sumo_speed_mps",
        "cpu_speed_mps",
        "gpu_speed_mps",
        "sumo_travel_time_s",
        "cpu_travel_time_s",
        "gpu_travel_time_s",
        "sumo_density_veh_per_m",
        "cpu_density_veh_per_m",
        "gpu_density_veh_per_m",
        "sumo_flow_veh_per_s",
        "cpu_flow_veh_per_s",
        "gpu_flow_veh_per_s",
        "abs_err_cpu_speed",
        "abs_err_gpu_speed",
        "abs_err_cpu_travel_time",
        "abs_err_gpu_travel_time",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    # SUMO가 차량을 관측한 edge(sampledSeconds>0)만 비교 — 미관측 edge는 0으로 채워져 있어
    # 엔진의 비제로 값과 비교하면 오차 통계가 왜곡됨.
    sumo_observed = {eid for eid, v in sumo_edge.items() if v.get("sampled_seconds", 0.0) > 0.0}
    edges = sorted(sumo_observed & set(cpu_edge.keys()) & set(gpu_edge.keys()))

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for eid in edges:
            s = sumo_edge[eid]
            c = cpu_edge[eid]
            g = gpu_edge[eid]
            w.writerow(
                {
                    "edge_id": eid,
                    "sumo_speed_mps": f"{s['speed_mps']:.6f}",
                    "cpu_speed_mps": f"{c['speed_mps']:.6f}",
                    "gpu_speed_mps": f"{g['speed_mps']:.6f}",
                    "sumo_travel_time_s": f"{s['travel_time_s']:.6f}",
                    "cpu_travel_time_s": f"{c['travel_time_s']:.6f}",
                    "gpu_travel_time_s": f"{g['travel_time_s']:.6f}",
                    "sumo_density_veh_per_m": f"{s['density_veh_per_m']:.6f}",
                    "cpu_density_veh_per_m": f"{c['density_veh_per_m']:.6f}",
                    "gpu_density_veh_per_m": f"{g['density_veh_per_m']:.6f}",
                    "sumo_flow_veh_per_s": f"{s['flow_veh_per_s']:.6f}",
                    "cpu_flow_veh_per_s": f"{c['flow_veh_per_s']:.6f}",
                    "gpu_flow_veh_per_s": f"{g['flow_veh_per_s']:.6f}",
                    "abs_err_cpu_speed": f"{abs(c['speed_mps'] - s['speed_mps']):.6f}",
                    "abs_err_gpu_speed": f"{abs(g['speed_mps'] - s['speed_mps']):.6f}",
                    "abs_err_cpu_travel_time": f"{abs(c['travel_time_s'] - s['travel_time_s']):.6f}",
                    "abs_err_gpu_travel_time": f"{abs(g['travel_time_s'] - s['travel_time_s']):.6f}",
                }
            )


def write_report(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Compare SUMO vs CPU-MT vs GPU with same conditions")
    p.add_argument("--python", default="python3")
    p.add_argument("--sumo", default="sumo")
    p.add_argument("--net-file", default="./map_import/gangnam4_fallback.net.xml")
    p.add_argument("--route-file", default="./map_import/vehicles_100k.sorted2.rou.xml")
    p.add_argument("--duration-sec", type=float, default=86400.0)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--cpu-workers", type=int, default=8)
    p.add_argument("--gpu-in-docker", action="store_true", default=True)
    p.add_argument("--no-gpu-in-docker", action="store_false", dest="gpu_in_docker")
    p.add_argument("--docker-image", default="gangnam4-cuda-sumo:latest")
    p.add_argument("--workspace", default="/home/mgkyung/ts")
    p.add_argument("--out-prefix", default="./map_import/compare_run")
    args = p.parse_args()

    net_file = Path(args.net_file)
    route_file = Path(args.route_file)
    if not net_file.exists():
        fail(f"net-file 없음: {net_file}")
    if not route_file.exists():
        fail(f"route-file 없음: {route_file}")

    edge_total, conn_total, edge_ext = count_net_stats(net_file)
    veh_n = count_vehicles(route_file)
    log(f"동일 조건 확인: edges_total={edge_total}, connections_total={conn_total}, edges_external={edge_ext}, vehicles={veh_n}")

    tag = time.strftime("%Y%m%d_%H%M%S")
    prefix = Path(f"{args.out_prefix}_{tag}")
    sumo_cfg = prefix.with_suffix(".sumocfg")
    sumo_trip = Path(f"{prefix}_sumo_tripinfo.xml")
    sumo_sum = Path(f"{prefix}_sumo_summary.xml")
    sumo_stat = Path(f"{prefix}_sumo_stat.xml")
    sumo_edge = Path(f"{prefix}_sumo_edge.xml")
    sumo_log = Path(f"{prefix}_sumo.log")
    cpu_csv = Path(f"{prefix}_cpu.csv")
    gpu_csv = Path(f"{prefix}_gpu.csv")
    report_csv = Path(f"{prefix}_report.csv")
    report_edge_csv = Path(f"{prefix}_report_edgewise.csv")

    # 1) SUMO
    net_rel = to_cfg_relative(net_file, sumo_cfg)
    route_rel = to_cfg_relative(route_file, sumo_cfg)
    sumo_sum_rel = to_cfg_relative(sumo_sum, sumo_cfg)
    sumo_trip_rel = to_cfg_relative(sumo_trip, sumo_cfg)
    sumo_stat_rel = to_cfg_relative(sumo_stat, sumo_cfg)
    sumo_edge_rel = to_cfg_relative(sumo_edge, sumo_cfg)

    sumo_cfg.write_text(
        "\n".join(
            [
                "<configuration>",
                "  <input>",
                f'    <net-file value="{net_rel}"/>',
                f'    <route-files value="{route_rel}"/>',
                "  </input>",
                "  <time>",
                '    <begin value="0"/>',
                f'    <end value="{int(args.duration_sec)}"/>',
                "  </time>",
                "  <output>",
                f'    <summary-output value="{sumo_sum_rel}"/>',
                f'    <tripinfo-output value="{sumo_trip_rel}"/>',
                f'    <statistic-output value="{sumo_stat_rel}"/>',
                f'    <edgedata-output value="{sumo_edge_rel}"/>',
                "  </output>",
                "</configuration>",
            ]
        ),
        encoding="utf-8",
    )
    t_sumo = run_cmd([args.sumo, "-c", str(sumo_cfg), "--no-warnings", "true", "--log", str(sumo_log)])

    # 2) CPU-MT
    t_cpu = run_cmd(
        [
            args.python,
            "./gangnam4_cuda/cpu_edge_simulator_mt.py",
            "--net-file",
            str(net_file),
            "--route-file",
            str(route_file),
            "--sim-duration",
            str(args.duration_sec),
            "--steps",
            str(args.steps),
            "--dt",
            str(args.dt),
            "--num-workers",
            str(args.cpu_workers),
            "--output-csv",
            str(cpu_csv),
        ]
    )

    # 3) GPU
    if args.gpu_in_docker:
        t_gpu = run_cmd(
            [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "-v",
                f"{args.workspace}:/workspace",
                "-w",
                "/workspace",
                args.docker_image,
                args.python,
                "./gangnam4_cuda/cuda_edge_simulator.py",
                "--net-file",
                str(net_file),
                "--route-file",
                str(route_file),
                "--sim-duration",
                str(args.duration_sec),
                "--steps",
                str(args.steps),
                "--dt",
                str(args.dt),
                "--output-csv",
                str(gpu_csv),
            ]
        )
    else:
        t_gpu = run_cmd(
            [
                args.python,
                "./gangnam4_cuda/cuda_edge_simulator.py",
                "--net-file",
                str(net_file),
                "--route-file",
                str(route_file),
                "--sim-duration",
                str(args.duration_sec),
                "--steps",
                str(args.steps),
                "--dt",
                str(args.dt),
                "--output-csv",
                str(gpu_csv),
            ]
        )

    # 4) 비교 요약
    sumo_trip_n, sumo_mean_dur, sumo_mean_speed = parse_sumo_tripinfo(sumo_trip)
    cpu_n, cpu_mean_speed, cpu_mean_tt = parse_edge_csv(cpu_csv)
    gpu_n, gpu_mean_speed, gpu_mean_tt = parse_edge_csv(gpu_csv)
    sumo_edge_map = parse_sumo_edgedata_last_interval(sumo_edge)
    cpu_edge_map = read_edge_metric_map(cpu_csv)
    gpu_edge_map = read_edge_metric_map(gpu_csv)
    write_edgewise_report(report_edge_csv, sumo_edge_map, cpu_edge_map, gpu_edge_map)

    rows = [
        {
            "engine": "SUMO",
            "edge_total": str(edge_total),
            "connection_total": str(conn_total),
            "edge_external": str(edge_ext),
            "vehicles": str(veh_n),
            "runtime_sec": f"{t_sumo:.3f}",
            "metric_count": str(sumo_trip_n),
            "mean_speed_mps": f"{sumo_mean_speed:.6f}",
            "mean_time_sec": f"{sumo_mean_dur:.6f}",
            "artifact": str(sumo_trip),
        },
        {
            "engine": "CPU_MT_EDGE",
            "edge_total": str(edge_total),
            "connection_total": str(conn_total),
            "edge_external": str(edge_ext),
            "vehicles": str(veh_n),
            "runtime_sec": f"{t_cpu:.3f}",
            "metric_count": str(cpu_n),
            "mean_speed_mps": f"{cpu_mean_speed:.6f}",
            "mean_time_sec": f"{cpu_mean_tt:.6f}",
            "artifact": str(cpu_csv),
        },
        {
            "engine": "GPU_EDGE",
            "edge_total": str(edge_total),
            "connection_total": str(conn_total),
            "edge_external": str(edge_ext),
            "vehicles": str(veh_n),
            "runtime_sec": f"{t_gpu:.3f}",
            "metric_count": str(gpu_n),
            "mean_speed_mps": f"{gpu_mean_speed:.6f}",
            "mean_time_sec": f"{gpu_mean_tt:.6f}",
            "artifact": str(gpu_csv),
        },
        {
            "engine": "EDGEWISE_1TO1",
            "edge_total": str(edge_total),
            "connection_total": str(conn_total),
            "edge_external": str(edge_ext),
            "vehicles": str(veh_n),
            "runtime_sec": "-",
            "metric_count": str(len(set(sumo_edge_map.keys()) & set(cpu_edge_map.keys()) & set(gpu_edge_map.keys()))),
            "mean_speed_mps": "-",
            "mean_time_sec": "-",
            "artifact": str(report_edge_csv),
        },
    ]
    write_report(report_csv, rows)
    log(f"비교 리포트 저장: {report_csv}")


if __name__ == "__main__":
    main()
