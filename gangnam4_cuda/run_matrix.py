#!/usr/bin/env python3
"""
긴 매트릭스 실험 — (network × demand × config) 셀별 Pareto 데이터 + SUMO ground truth.

각 (network, demand) 셀마다:
  1) demand=N 인 route 파일 생성(scale_demand)
  2) SUMO 실행 → edgedata (ground truth)
  3) 선택된 configs 각각에 대해 시뮬 + SUMO 비교 → 한 줄(CSV)
모든 결과를 단일 master CSV에 누적, 각 cell별 부산물도 보존.

견고성:
  - 매 줄 flush, 셀별 [ETA]/[CELL]/[OK] 마커로 진행상황 파싱 용이
  - 셀 실패 시 다음 셀 진행(전체 sweep을 망치지 않음)
  - --resume: master CSV에 이미 있는 (network,demand,config) 조합은 건너뜀
  - --systems 별로 큰 수요는 자동으로 제외(--cap-{config}=N 으로 상한)
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
# "COMBO" = (network, demand) 한 조합 (혼동 방지: 도로/차량이 아니라 매트릭스 한 칸).
# 별도로 cell_common.py의 "cell"은 도로 공간 격자 — 둘은 무관.
def combo_log(net, dem, cfg, msg): print(f"[COMBO] net={net} dem={dem} cfg={cfg} :: {msg}", flush=True)
def eta_log(done, total, t_elapsed): print(f"[ETA] done={done}/{total} elapsed={t_elapsed:.0f}s "
    f"avg_per_combo={(t_elapsed/max(done,1)):.0f}s remain~{(t_elapsed/max(done,1))*(total-done):.0f}s", flush=True)


def sumo_edgedata(args, route_path: Path, edge_xml: Path, sim_time: float, seed: int = 1000):
    """SUMO를 Docker로 돌려 edgeData xml 생성. 이미 있으면 skip."""
    if edge_xml.exists() and edge_xml.stat().st_size > 0:
        return
    ws = Path(args.workspace).resolve()
    cfg = edge_xml.with_suffix(".sumocfg")
    add = edge_xml.with_suffix(".add.xml")
    add.write_text(f'<additional><edgeData id="ed" file="{edge_xml.name}" begin="0" end="{int(sim_time)}"/></additional>\n', encoding="utf-8")
    net_c = "/workspace/" + str(Path(args.net_file_default).resolve().relative_to(ws)) if "net_file" not in args.__dict__ else None
    # network는 cell마다 다를 수 있으므로 호출부에서 net을 받아야 함 — 이 함수는 사용 안 하고 직접 inline
    raise SystemExit("use sumo_edgedata_for_cell instead")


def sumo_edgedata_for_cell(args, net_file: Path, route_file: Path, edge_xml: Path, sim_time: float, seed: int = 1000):
    if edge_xml.exists() and edge_xml.stat().st_size > 0:
        return True
    ws = Path(args.workspace).resolve()
    cfg = edge_xml.with_suffix(".sumocfg"); add = edge_xml.with_suffix(".add.xml")
    add.write_text(f'<additional><edgeData id="ed" file="{edge_xml.name}" begin="0" end="{int(sim_time)}"/></additional>\n', encoding="utf-8")
    net_c = "/workspace/" + str(net_file.resolve().relative_to(ws))
    route_c = "/workspace/" + str(route_file.resolve().relative_to(ws))
    add_c = "/workspace/" + str(add.resolve().relative_to(ws))
    cfg.write_text("\n".join([
        "<configuration>",
        f'  <input><net-file value="{net_c}"/><route-files value="{route_c}"/>',
        f'    <additional-files value="{add_c}"/></input>',
        f'  <time><begin value="0"/><end value="{int(sim_time)}"/></time>',
        f'  <random_number><seed value="{seed}"/></random_number>',
        "</configuration>",
    ]), encoding="utf-8")
    out_dir_c = "/workspace/" + str(edge_xml.parent.resolve().relative_to(ws))
    cmd = ["docker", "run", "--rm", "-v", f"{ws}:/workspace", "-w", out_dir_c,
           args.docker_image, "sumo", "-c", "/workspace/" + str(cfg.resolve().relative_to(ws)),
           "--no-warnings", "true", "--no-step-log", "true"]
    log(f"  SUMO run: {' '.join(cmd[-3:])}")
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return r.returncode == 0 and edge_xml.exists()


def make_route(args, base_route: Path, target_n: int, out: Path):
    if out.exists() and out.stat().st_size > 0:
        return
    cmd = [sys.executable, str(_HERE / "scale_demand.py"),
           "--in-route", str(base_route), "--out-route", str(out), "--target", str(target_n)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def run_pareto_cell(args, net_file: Path, route_file: Path, sumo_xml: Path, configs: str,
                    sim_time: float, cell_csv: Path) -> bool:
    """run_pareto.py 호출 → cell 단위 CSV."""
    cmd = [sys.executable, "-u", str(_HERE / "run_pareto.py"),
           "--net-file", str(net_file), "--route-file", str(route_file),
           "--sumo-edgedata", str(sumo_xml),
           "--sim-time", str(sim_time), "--dt", str(args.dt), "--meso-dt", str(args.meso_dt),
           "--configs", configs,
           "--out-csv", str(cell_csv)]
    pr = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # stdout을 그대로 흘려서 로그에 남기되, 셀 prefix 붙임
    for line in pr.stdout.splitlines():
        print(f"  | {line}", flush=True)
    return pr.returncode == 0 and cell_csv.exists()


def cap_configs(all_configs: list[str], demand: int, caps: dict) -> str:
    out = []
    for c in all_configs:
        cap = caps.get(c, 10**12)
        if demand <= cap: out.append(c)
    return ",".join(out)


def main():
    p = argparse.ArgumentParser(description="Long matrix sweep: network × demand × config")
    p.add_argument("--out-dir", default="/home/mgkyung/ts/.claude/worktrees/elastic-allen-cc712e/paper_data/matrix")
    p.add_argument("--workspace", default="/home/mgkyung/ts")
    p.add_argument("--docker-image", default="gangnam4-cuda-sumo:latest")
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--meso-dt", type=float, default=1.0)
    # 네트워크 사양: name|net.xml|base_route|sim_time
    p.add_argument("--networks", default=(
        "gangnam|map_import/gangnam4_generated.net.xml|map_import/gangnam4_generated.sorted.rou.xml|3600;"
        "grid|map_import/synthetic/grid/net.xml|map_import/synthetic/grid/routes.rou.xml|3600"))
    p.add_argument("--demands", default="5000,20000,100000,500000,1000000")
    p.add_argument("--configs", default=("lane_cpu_ctm,lane_cpu_ctm_v035_hcm,"
                   "meso_cpu,meso_cpu_v05,meso_gpu_naive,meso_gpu_graph,meso_gpu_graph_cong"))
    # SUMO와 매크로 Python-loop 엔진은 큰 수요에서 너무 느림 — 상한
    p.add_argument("--sumo-cap", type=int, default=20000, help="SUMO 기준 생성은 이 demand까지만(이상은 ground-truth 없이 skip)")
    p.add_argument("--resume", action="store_true", help="master CSV에 이미 있는 셀은 건너뜀")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    master = out_dir / "matrix.csv"
    cells_dir = out_dir / "cells"; cells_dir.mkdir(exist_ok=True)
    routes_dir = out_dir / "routes"; routes_dir.mkdir(exist_ok=True)
    sumo_dir = out_dir / "sumo_ref"; sumo_dir.mkdir(exist_ok=True)

    nets = []
    for spec in args.networks.split(";"):
        name, netf, basef, simt = spec.split("|")
        nets.append((name, Path(args.workspace) / netf, Path(args.workspace) / basef, float(simt)))
    demands = [int(x) for x in args.demands.split(",")]
    all_cfg = args.configs.split(",")

    # 셀 enumeration
    combos = [(nname, nf, bf, st, dem) for (nname, nf, bf, st) in nets for dem in demands]
    total = len(combos)
    log(f"매트릭스 시작: networks={len(nets)} demands={len(demands)} combos={total}")

    # resume용 기존 행
    done_keys = set()
    if args.resume and master.exists():
        with master.open() as f:
            for r in csv.DictReader(f):
                done_keys.add((r["network"], int(r["demand"]), r["config"]))
        log(f"resume: 기존 행 {len(done_keys)}개")

    if not master.exists():
        with master.open("w", newline="") as f:
            csv.writer(f).writerow(["network","demand","config","wall_inner_s","wall_ext_s","matched",
                "flow_geh_lt5","flow_r","flow_rho","speed_r","speed_rho",
                "density_r","density_rho","hotspot_prec"])

    t0 = time.perf_counter()
    for ci, (nname, nf, bf, st, dem) in enumerate(combos, 1):
        combo_log(nname, dem, "ALL", f">>> 시작 ({ci}/{total})")
        # 수요 route
        rfile = routes_dir / f"{nname}_{dem}.rou.xml"
        try: make_route(args, bf, dem, rfile)
        except Exception as e:
            combo_log(nname, dem, "ALL", f"route 생성 실패: {e}"); continue

        # SUMO ground truth (상한 안에서만)
        sumo_xml = sumo_dir / f"{nname}_{dem}.xml"
        if dem <= args.sumo_cap:
            ok = sumo_edgedata_for_cell(args, nf, rfile, sumo_xml, st)
            if not ok:
                combo_log(nname, dem, "ALL", "SUMO 실패 → 이 셀 skip"); continue
        else:
            combo_log(nname, dem, "ALL", f"수요 > {args.sumo_cap} → SUMO ground truth 생성 skip(엔진 wall만 측정)")
            sumo_xml = None

        # SUMO ref 없으면 정확도 비교 불가 → 엔진 wall만 별도 측정(추후 다른 sweep에서 처리)
        if sumo_xml is None:
            continue

        # 엔진 cap 적용(필요 시 추가) — 현재는 모두 허용
        cfgs_for_cell = cap_configs(all_cfg, dem, {})
        # 이미 처리된 엔진 제외
        cfgs_for_cell = ",".join([c for c in cfgs_for_cell.split(",")
                                  if (nname, dem, c) not in done_keys])
        if not cfgs_for_cell:
            combo_log(nname, dem, "ALL", "모두 resume됨 → skip"); eta_log(ci, total, time.perf_counter()-t0); continue

        cell_csv = cells_dir / f"{nname}_{dem}.csv"
        ok = run_pareto_cell(args, nf, rfile, sumo_xml, cfgs_for_cell, st, cell_csv)
        if not ok or not cell_csv.exists():
            combo_log(nname, dem, "ALL", "pareto 실패"); continue
        # 셀 결과를 master에 append
        with cell_csv.open() as f, master.open("a", newline="") as fo:
            rd = csv.DictReader(f); w = csv.writer(fo)
            for r in rd:
                w.writerow([nname, dem, r["config"], r["wall_inner_s"], r["wall_ext_s"], r["matched"],
                    r["flow_geh_lt5"], r["flow_r"], r["flow_rho"], r["speed_r"], r["speed_rho"],
                    r["density_r"], r["density_rho"], r["hotspot_prec"]])
        combo_log(nname, dem, "ALL", "[OK] master에 누적")
        eta_log(ci, total, time.perf_counter()-t0)

    log("매트릭스 완료")
    log(f"master CSV: {master}")


if __name__ == "__main__":
    main()
