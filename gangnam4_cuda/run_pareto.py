#!/usr/bin/env python3
"""
E1 Pareto 데이터 생성기 — (시스템 × calibration) 조합별 (wall, accuracy) 점.

Figure 1 입력: 가로축 wall(s, log), 세로축 정확도(flow GEH<5 또는 Spearman).
각 점은 한 (engine config) → 1 시뮬 + SUMO 비교.

엔진 config 예시:
  edge_cpu_lwr            (gen-1 baseline)
  lane_cpu_lwr            (gen-2 LWR)
  lane_cpu_ctm            (gen-2 CTM, uncalibrated)
  lane_cpu_ctm_v035       (vmax-scale=0.35)
  lane_cpu_ctm_v035_hcm   (+ HCM 0.7/0.5)
  meso_cpu                (gen-3 meso, no calib)
  meso_cpu_v05            (vmax-scale=0.5)
  meso_gpu_naive          (gen-3 GPU naive)
  meso_gpu_graph          (gen-3 GPU branch-free + CUDA graph)
  meso_gpu_graph_cong     (+ congestion-dependent junction delay)

SUMO는 reference. 모든 비교는 동일 net+route+sim_time, 동일 SUMO edgedata XML 기준.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_WALL = re.compile(r"완료:\s*([0-9.]+)s")


def log(m): print(f"[LOG] {m}", flush=True)


def run_cmd(cmd, env=None):
    """Return (wall_inner_s, full_stdout)."""
    pr = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    if pr.returncode != 0:
        print(pr.stdout[-2000:])
        raise SystemExit(f"실패: {' '.join(cmd[:6])}...")
    m = _WALL.search(pr.stdout)
    return (float(m.group(1)) if m else float("nan")), pr.stdout


def score_vs_sumo(edge_csv: Path, sumo_xml: Path) -> dict:
    """run_compare_sumo로 평가, 요약 dict 반환."""
    out_prefix = f"/tmp/pareto_cmp_{int(time.time()*1000)}"
    cmd = [sys.executable, str(_HERE / "run_compare_sumo.py"),
           "--engine-edge-csv", str(edge_csv),
           "--sumo-edgedata", str(sumo_xml),
           "--topk", "100", "--out-prefix", out_prefix]
    pr = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = pr.stdout
    def grab(rx, txt=out):
        m = re.search(rx, txt)
        return float(m.group(1)) if m else float("nan")
    return {
        "matched": int(grab(r"매칭 edge 수:\s*(\d+)")),
        "flow_r": grab(r"flow_veh_per_s\s+r=([\-0-9.]+)"),
        "flow_rho": grab(r"flow_veh_per_s\s+r=[\-0-9.]+\s+rho=([\-0-9.]+)"),
        "flow_geh_lt5": grab(r"flow_veh_per_s\s.*GEH<5=([0-9.]+)%") / 100.0,
        "speed_r": grab(r"speed_mps\s+r=([\-0-9.]+)"),
        "speed_rho": grab(r"speed_mps\s+r=[\-0-9.]+\s+rho=([\-0-9.]+)"),
        "density_r": grab(r"density_veh_per_m\s+r=([\-0-9.]+)"),
        "density_rho": grab(r"density_veh_per_m\s+r=[\-0-9.]+\s+rho=([\-0-9.]+)"),
        "hotspot_prec": grab(r"hotspot.*precision=([0-9.]+)"),
    }


def build_run(args, name: str, tmp: Path) -> list[str] | None:
    """엔진 config 이름 → 실행 명령."""
    net, route, st, dt_macro, dt_meso = args.net_file, args.route_file, args.sim_time, args.dt, args.meso_dt
    out = tmp / f"{name}.edge.csv"
    if name == "edge_cpu_lwr":
        return [sys.executable, str(_HERE / "cpu_edge_simulator_mt.py"),
                "--net-file", net, "--route-file", route,
                "--steps", str(int(st/dt_macro)), "--dt", str(dt_macro),
                "--log-interval", "999999", "--output-csv", str(out)]
    if name.startswith("lane_cpu_"):
        cmd = [sys.executable, str(_HERE / "lane_cpu_simulator_mt.py"),
               "--net-file", net, "--route-file", route,
               "--sim-time", str(st), "--dt", str(dt_macro), "--time-average",
               "--log-interval", "999999", "--output-csv", str(tmp / f"{name}.lane.csv"),
               "--edge-output-csv", str(out)]
        if name == "lane_cpu_lwr":
            cmd += ["--model", "lwr"]
        elif name == "lane_cpu_ctm":
            cmd += ["--model", "ctm"]
        elif name == "lane_cpu_ctm_v035":
            cmd += ["--model", "ctm", "--vmax-scale", "0.35"]
        elif name == "lane_cpu_ctm_v035_hcm":
            cmd += ["--model", "ctm", "--vmax-scale", "0.35",
                    "--major-left-factor", "0.7", "--minor-factor", "0.5"]
        return cmd
    if name.startswith("meso_cpu"):
        cmd = [sys.executable, str(_HERE / "meso_sim.py"),
               "--net-file", net, "--route-file", route,
               "--sim-time", str(st), "--dt", str(dt_meso),
               "--log-interval", "999999", "--trip-output-csv", "",
               "--edge-output-csv", str(out)]
        if name == "meso_cpu_v05":
            cmd += ["--vmax-scale", "0.5"]
        return cmd
    # GPU 엔진(docker) 별도 처리는 main에서
    return None


def build_gpu_run(args, name: str, tmp: Path) -> list[str] | None:
    ws = Path(args.workspace).resolve()
    def cp(p): return "/workspace/" + str(Path(p).resolve().relative_to(ws))
    net_c, route_c = cp(args.net_file), cp(args.route_file)
    out_c = f"/workspace/map_import/_pareto_{name}.edge.csv"
    common = (f"--net-file {net_c} --route-file {route_c} "
              f"--sim-time {args.sim_time} --dt {args.meso_dt} ")
    if name == "meso_gpu_naive":
        inner = (f"python3 /workspace/gangnam4_cuda/meso_gpu.py {common}"
                 f"--log-interval 999999 --trip-output-csv '' --edge-output-csv {out_c}")
    elif name == "meso_gpu_graph":
        inner = (f"python3 /workspace/gangnam4_cuda/meso_gpu_graph.py {common}"
                 f"--edge-output-csv {out_c}")
    elif name == "meso_gpu_graph_cong":
        inner = (f"python3 /workspace/gangnam4_cuda/meso_gpu_graph.py {common}"
                 f"--junction-cong-coef 10 --edge-output-csv {out_c}")
    else:
        return None
    return ["docker", "run", "--rm", "--gpus", "all",
            "-v", f"{ws}:/workspace", "-v", f"{_HERE}:/workspace/gangnam4_cuda",
            "-w", "/workspace", args.docker_image, "bash", "-lc", inner]


def main():
    p = argparse.ArgumentParser(description="Pareto data: (system × calibration) → (wall, accuracy)")
    p.add_argument("--net-file", default="./map_import/gangnam4_generated.net.xml")
    p.add_argument("--route-file", default="./map_import/gangnam4_generated.sorted.rou.xml")
    p.add_argument("--sumo-edgedata", required=True, help="기준 SUMO edgedata xml")
    p.add_argument("--sim-time", type=float, default=3600.0)
    p.add_argument("--dt", type=float, default=0.5, help="macro/CTM dt")
    p.add_argument("--meso-dt", type=float, default=1.0)
    p.add_argument("--configs", default=("edge_cpu_lwr,lane_cpu_lwr,lane_cpu_ctm,"
                   "lane_cpu_ctm_v035,lane_cpu_ctm_v035_hcm,meso_cpu,meso_cpu_v05,"
                   "meso_gpu_naive,meso_gpu_graph,meso_gpu_graph_cong"))
    p.add_argument("--workspace", default="/home/mgkyung/ts")
    p.add_argument("--docker-image", default="gangnam4-cuda-sumo:latest")
    p.add_argument("--tmp-dir", default="/tmp/pareto")
    p.add_argument("--out-csv", default="/tmp/pareto/pareto.csv")
    args = p.parse_args()

    tmp = Path(args.tmp_dir); tmp.mkdir(parents=True, exist_ok=True)
    sumo_xml = Path(args.sumo_edgedata)
    if not sumo_xml.exists():
        raise SystemExit(f"SUMO edgedata 없음: {sumo_xml}")

    rows = []
    for name in args.configs.split(","):
        log(f"--- {name} ---")
        cpu_cmd = build_run(args, name, tmp)
        gpu_cmd = build_gpu_run(args, name, tmp) if cpu_cmd is None else None
        if cpu_cmd is None and gpu_cmd is None:
            log(f"  unknown config, skip"); continue
        cmd = cpu_cmd or gpu_cmd
        t0 = time.perf_counter()
        wall_inner, _ = run_cmd(cmd)
        wall_ext = time.perf_counter() - t0

        # GPU 결과는 컨테이너가 /workspace에 쓴 경로를 호스트에서 매핑
        if gpu_cmd is not None:
            edge_csv = Path(f"/home/mgkyung/ts/map_import/_pareto_{name}.edge.csv")
        else:
            edge_csv = tmp / f"{name}.edge.csv"

        if not edge_csv.exists():
            log(f"  edge csv 없음: {edge_csv}, skip 점수"); continue
        s = score_vs_sumo(edge_csv, sumo_xml)
        s.update({"config": name, "wall_inner_s": wall_inner, "wall_ext_s": wall_ext})
        rows.append(s)
        log(f"  wall={wall_inner:.2f}s GEH<5={s['flow_geh_lt5']*100:.1f}% "
            f"flow_rho={s['flow_rho']:.3f} speed_rho={s['speed_rho']:.3f}")

    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["config", "wall_inner_s", "wall_ext_s", "matched",
            "flow_geh_lt5", "flow_r", "flow_rho",
            "speed_r", "speed_rho", "density_r", "density_rho", "hotspot_prec"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]) for k in cols})
    log("=" * 88)
    log(f"{'config':28s}  {'wall_s':>8s}  {'GEH<5':>6s}  {'flow_ρ':>7s}  {'speed_ρ':>8s}  {'dens_ρ':>7s}")
    for r in rows:
        log(f"{r['config']:28s}  {r['wall_inner_s']:8.2f}  {r['flow_geh_lt5']*100:5.1f}%  "
            f"{r['flow_rho']:7.3f}  {r['speed_rho']:8.3f}  {r['density_rho']:7.3f}")
    log("=" * 88)
    log(f"Pareto data 저장: {out}")


if __name__ == "__main__":
    main()
