#!/usr/bin/env python3
"""
Multi-seed SUMO 실행 + 집계 — 논문용 ground truth(평균±CI) 및 noise floor.

SUMO는 확률적(car-following sigma, lane-change, departLane/Speed)이므로 단일 실행은
표본 1개일 뿐이다. N개 seed로 실행해:
  1) edge별 (speed/density/flow) 평균·표준편차·95% CI 집계 → ground truth
  2) **noise floor**: seed쌍 간 SUMO-vs-SUMO 일치도(flow GEH<5, speed Spearman) 평균.
     어떤 엔진도 단일 SUMO와 이보다 잘 맞출 수 없다(상한선) — 논문 해석의 기준.

출력:
  <prefix>_sumo_mean.edge.csv   엔진 스키마(veh/m, veh/s) 평균 — run_compare_sumo --ref-edge-csv 로 바로 사용
  <prefix>_sumo_stats.csv       edge별 mean/std/ci/coverage
  <prefix>_noisefloor.csv       seed쌍 cross-consistency 요약

SUMO는 Docker 이미지에서 실행(로컬 미설치). edgeData additional로 구간 평균 수집.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from run_compare_sumo_cpu_gpu import parse_sumo_edgedata_last_interval  # 단위변환+sampled 포함
import metrics as M


def log(m): print(f"[LOG] {m}")
def fail(m): print(f"[ERROR] {m}"); raise SystemExit(1)


def run_one_seed(args, seed: int, edgedata_host: Path) -> None:
    """seed 1개에 대해 SUMO를 Docker에서 실행, edgeData xml 생성."""
    ws = Path(args.workspace).resolve()
    # 컨테이너 내부 경로로 변환 (host workspace ↔ /workspace)
    def cpath(p: Path) -> str:
        return "/workspace/" + str(p.resolve().relative_to(ws))

    net_c = cpath(Path(args.net_file))
    route_c = cpath(Path(args.route_file))
    edge_c = cpath(edgedata_host)
    add_host = edgedata_host.with_suffix(".add.xml")
    cfg_host = edgedata_host.with_suffix(".sumocfg")
    add_c = cpath(add_host)
    cfg_c = cpath(cfg_host)

    add_host.write_text("\n".join([
        "<additional>",
        f'  <edgeData id="ed" file="{Path(edge_c).name}" begin="0" end="{int(args.duration_sec)}"/>',
        "</additional>",
    ]), encoding="utf-8")
    # edgeData file은 cfg 기준 상대경로로 쓰이므로, 컨테이너 작업디렉토리를 edgedata 폴더로 맞춤
    cfg_host.write_text("\n".join([
        "<configuration>",
        "  <input>",
        f'    <net-file value="{net_c}"/>',
        f'    <route-files value="{route_c}"/>',
        f'    <additional-files value="{add_c}"/>',
        "  </input>",
        f'  <time><begin value="0"/><end value="{int(args.duration_sec)}"/></time>',
        f'  <random_number><seed value="{seed}"/></random_number>',
        "</configuration>",
    ]), encoding="utf-8")

    # edgeData의 file 경로는 add.xml 위치 기준 상대 → add.xml과 같은 폴더에 edge xml 생성되도록
    # SUMO는 output을 cfg의 작업dir 기준으로 쓴다. -w 를 edgedata 폴더로.
    out_dir_c = "/workspace/" + str(edgedata_host.parent.resolve().relative_to(ws))
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{ws}:/workspace", "-w", out_dir_c,
        args.docker_image,
        "sumo", "-c", cfg_c, "--no-warnings", "true", "--no-step-log", "true",
        "--seed", str(seed),
    ]
    log(f"seed={seed} CMD: {' '.join(cmd)}")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        fail(f"SUMO 실행 실패(seed={seed}, rc={rc})")
    if not edgedata_host.exists():
        fail(f"edgeData 미생성: {edgedata_host}")


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-seed SUMO ground truth + noise floor")
    p.add_argument("--net-file", default="./map_import/gangnam4_generated.net.xml")
    p.add_argument("--route-file", default="./map_import/gangnam4_generated.sorted.rou.xml")
    p.add_argument("--duration-sec", type=float, default=3600.0)
    p.add_argument("--seeds", type=int, default=10, help="seed 개수(1..N)")
    p.add_argument("--seed-base", type=int, default=1000)
    p.add_argument("--workspace", default="/home/mgkyung/ts")
    p.add_argument("--docker-image", default="gangnam4-cuda-sumo:latest")
    p.add_argument("--out-prefix", default="/home/mgkyung/ts/map_import/sumo_seeds/run")
    p.add_argument("--min-coverage", type=float, default=0.5,
                   help="edge가 평균에 포함되려면 sampled 되어야 하는 seed 비율(>=)")
    p.add_argument("--reuse", action="store_true", help="기존 edgeData xml 있으면 재실행 생략")
    args = p.parse_args()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed_base + i for i in range(args.seeds)]
    # 1) 각 seed 실행 + 파싱
    per_seed_maps: list[dict] = []
    for s in seeds:
        edl = Path(f"{out_prefix}_seed{s}_edge.xml")
        if not (args.reuse and edl.exists()):
            run_one_seed(args, s, edl)
        m = parse_sumo_edgedata_last_interval(edl)  # 단위변환됨(veh/m, veh/s), sampled_seconds 포함
        sampled = {eid: v for eid, v in m.items() if v.get("sampled_seconds", 0) > 0}
        per_seed_maps.append(sampled)
        log(f"seed={s}: sampled edges={len(sampled)}")

    N = len(per_seed_maps)
    if N == 0:
        fail("seed 결과 0개")

    # 2) edge별 집계 (coverage >= min-coverage 인 edge만)
    all_edges = set().union(*[set(m) for m in per_seed_maps])
    keys = ["speed_mps", "density_veh_per_m", "flow_veh_per_s"]
    stats_rows = []
    mean_rows = []
    t95 = 2.262 if N == 10 else 1.96  # 대략(N=10 t), 그 외 정규근사
    kept = 0
    for eid in sorted(all_edges):
        present = [m[eid] for m in per_seed_maps if eid in m]
        cov = len(present) / N
        if cov < args.min_coverage:
            continue
        kept += 1
        row = {"edge_id": eid, "coverage": cov, "n_seeds": len(present)}
        for k in keys:
            vals = np.array([pp[k] for pp in present], float)
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
            ci = t95 * std / math.sqrt(vals.size) if vals.size > 1 else 0.0
            row[f"{k}_mean"] = mean
            row[f"{k}_std"] = std
            row[f"{k}_ci95"] = ci
        stats_rows.append(row)
        mean_rows.append([eid, "", row["density_veh_per_m_mean"],
                          row["speed_mps_mean"], row["flow_veh_per_s_mean"]])

    log(f"집계 edge={kept}/{len(all_edges)} (coverage>={args.min_coverage}), seeds={N}")

    # mean CSV (엔진 스키마 — run_compare_sumo --ref-edge-csv 로 사용)
    mean_csv = Path(f"{out_prefix}_sumo_mean.edge.csv")
    with mean_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["edge_id", "lanes", "density_veh_per_m", "speed_mps", "flow_veh_per_s"])
        w.writerows(mean_rows)
    log(f"평균 edge CSV: {mean_csv}")

    stats_csv = Path(f"{out_prefix}_sumo_stats.csv")
    with stats_csv.open("w", newline="") as f:
        cols = (["edge_id", "coverage", "n_seeds"]
                + [f"{k}_{s}" for k in keys for s in ("mean", "std", "ci95")])
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(stats_rows)
    log(f"통계 CSV: {stats_csv}")

    # 3) Noise floor: seed쌍 간 cross-consistency (공통 sampled edge 위에서)
    if N >= 2:
        flow_geh = []; speed_rho = []; flow_rho = []; dens_rho = []
        for i in range(N):
            for j in range(i + 1, N):
                common = sorted(set(per_seed_maps[i]) & set(per_seed_maps[j]))
                if len(common) < 10:
                    continue
                fi = np.array([per_seed_maps[i][e]["flow_veh_per_s"] for e in common]) * 3600
                fj = np.array([per_seed_maps[j][e]["flow_veh_per_s"] for e in common]) * 3600
                si = np.array([per_seed_maps[i][e]["speed_mps"] for e in common])
                sj = np.array([per_seed_maps[j][e]["speed_mps"] for e in common])
                di = np.array([per_seed_maps[i][e]["density_veh_per_m"] for e in common])
                dj = np.array([per_seed_maps[j][e]["density_veh_per_m"] for e in common])
                flow_geh.append(M.geh_fraction(fi, fj, 5.0))
                speed_rho.append(M.spearman_rho(si, sj))
                flow_rho.append(M.spearman_rho(fi, fj))
                dens_rho.append(M.spearman_rho(di, dj))

        def ms(x): a = np.array(x, float); return (float(a.mean()), float(a.std()))
        fg_m, fg_s = ms(flow_geh); sr_m, sr_s = ms(speed_rho)
        fr_m, fr_s = ms(flow_rho); dr_m, dr_s = ms(dens_rho)
        log("=" * 64)
        log(f"NOISE FLOOR (SUMO seed쌍 {len(flow_geh)}쌍, 상한선):")
        log(f"  flow GEH<5  = {fg_m*100:.1f}% ± {fg_s*100:.1f}")
        log(f"  flow  rho   = {fr_m:.4f} ± {fr_s:.4f}")
        log(f"  speed rho   = {sr_m:.4f} ± {sr_s:.4f}")
        log(f"  density rho = {dr_m:.4f} ± {dr_s:.4f}")
        log("=" * 64)
        nf_csv = Path(f"{out_prefix}_noisefloor.csv")
        with nf_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "mean", "std", "n_pairs"])
            w.writerow(["flow_geh_lt5", f"{fg_m:.6f}", f"{fg_s:.6f}", len(flow_geh)])
            w.writerow(["flow_spearman", f"{fr_m:.6f}", f"{fr_s:.6f}", len(flow_rho)])
            w.writerow(["speed_spearman", f"{sr_m:.6f}", f"{sr_s:.6f}", len(speed_rho)])
            w.writerow(["density_spearman", f"{dr_m:.6f}", f"{dr_s:.6f}", len(dens_rho)])
        log(f"noise floor CSV: {nf_csv}")


if __name__ == "__main__":
    main()
