#!/usr/bin/env python3
"""
스케일링 벤치마크(E2) — wall-time / throughput vs 차량 수(고정 네트워크).

핵심 가설:
  - lane-CTM 비용 = O(edges × steps), **차량 수와 무관**(수요는 부팅 시 1회 반영).
  - meso 비용 = O(vehicles × steps), 차량 수에 선형.
  - SUMO(미시) = O(vehicles), 가장 느림.
이 대비가 "대규모에서 매크로 CTM의 우위" 주장의 근거(Figure 2).

각 차량 수에 대해 scale_demand로 route를 만들고, 지정 엔진들을 subprocess로 실행,
완료 로그의 wall-time을 파싱하여 CSV로 기록. GPU는 별도(docker)로 측정 권장.
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
_WALL_RE = re.compile(r"완료:\s*([0-9.]+)s")


def log(m): print(f"[LOG] {m}")


def run_capture(cmd: list[str]) -> tuple[float, float]:
    """subprocess 실행, (로그파싱 wall, 외부측정 wall) 반환."""
    t0 = time.perf_counter()
    pr = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    ext = time.perf_counter() - t0
    inner = float("nan")
    for line in pr.stdout.splitlines():
        m = _WALL_RE.search(line)
        if m:
            inner = float(m.group(1))
    if pr.returncode != 0:
        print(pr.stdout[-2000:])
        raise SystemExit(f"엔진 실행 실패: {' '.join(cmd[:6])}...")
    return inner, ext


def main():
    p = argparse.ArgumentParser(description="Scaling benchmark across vehicle counts")
    p.add_argument("--net-file", default="./map_import/gangnam4_generated.net.xml")
    p.add_argument("--base-route", default="./map_import/gangnam4_generated.sorted.rou.xml")
    p.add_argument("--counts", default="10000,50000,100000,250000,500000",
                   help="콤마구분 차량 수")
    p.add_argument("--engines", default="meso,lane_cpu_ctm",
                   help="meso | lane_cpu_ctm | lane_cpu_lwr 중 콤마구분")
    p.add_argument("--sim-time", type=float, default=3600.0)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--tmp-dir", default="/tmp/scaling")
    p.add_argument("--out-csv", default="/tmp/scaling/scaling_results.csv")
    args = p.parse_args()

    tmp = Path(args.tmp_dir); tmp.mkdir(parents=True, exist_ok=True)
    counts = [int(c) for c in args.counts.split(",")]
    engines = args.engines.split(",")
    rows = []

    for n in counts:
        route = tmp / f"route_{n}.rou.xml"
        log(f"--- 차량 {n} 수요 생성 ---")
        subprocess.run([sys.executable, str(_HERE / "scale_demand.py"),
                        "--in-route", args.base_route, "--out-route", str(route),
                        "--target", str(n)], check=True)
        for eng in engines:
            if eng == "meso":
                cmd = [sys.executable, str(_HERE / "meso_sim.py"),
                       "--net-file", args.net_file, "--route-file", str(route),
                       "--sim-time", str(args.sim_time), "--dt", str(args.dt),
                       "--log-interval", "999999", "--trip-output-csv", "",
                       "--edge-output-csv", str(tmp / f"meso_{n}.edge.csv")]
            elif eng in ("lane_cpu_ctm", "lane_cpu_lwr"):
                model = "ctm" if eng.endswith("ctm") else "lwr"
                cmd = [sys.executable, str(_HERE / "lane_cpu_simulator_mt.py"),
                       "--net-file", args.net_file, "--route-file", str(route),
                       "--model", model, "--sim-time", str(args.sim_time), "--dt", str(args.dt),
                       "--log-interval", "999999",
                       "--output-csv", str(tmp / f"{eng}_{n}.csv"),
                       "--edge-output-csv", str(tmp / f"{eng}_{n}.edge.csv")]
            else:
                log(f"미지원 엔진 스킵: {eng}"); continue
            inner, ext = run_capture(cmd)
            steps = int(round(args.sim_time / args.dt))
            thru = (n * steps) / inner if inner > 0 else float("nan")
            rows.append({"engine": eng, "n_veh": n, "wall_s": f"{inner:.3f}",
                         "ext_wall_s": f"{ext:.3f}", "veh_steps_per_s": f"{thru:.0f}"})
            log(f"  {eng:14s} n={n:>7d}  wall={inner:.3f}s  ext={ext:.1f}s")

    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["engine", "n_veh", "wall_s", "ext_wall_s", "veh_steps_per_s"])
        w.writeheader(); w.writerows(rows)
    log(f"스케일링 결과 저장: {out}")
    # 요약 표
    log("=" * 56)
    for eng in engines:
        es = [r for r in rows if r["engine"] == eng]
        if es:
            log(f"{eng}:  " + "  ".join(f"{r['n_veh']}={r['wall_s']}s" for r in es))
    log("=" * 56)


if __name__ == "__main__":
    main()
