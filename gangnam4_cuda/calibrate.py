#!/usr/bin/env python3
"""
Held-out 캘리브레이션(E3) — "테스트셋에 튜닝했다" 비판 방어.

train 시나리오에서 (vmax_scale, major_left_factor, minor_factor) 그리드 탐색으로
flow RMSN을 최소화 → 최적 파라미터 동결 → **held-out test 시나리오**에서 평가.
train↔test 정확도 격차(gap)가 작으면 캘리브레이션이 일반화됨을 보인다.

엔진은 lane CTM(CPU, host 실행 — docker 불필요). ground truth는 edge-keyed CSV
(예: multi-seed 평균) 또는 SUMO edgedata xml.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import metrics as M
from run_compare_sumo_cpu_gpu import parse_sumo_edgedata_last_interval


def log(m): print(f"[LOG] {m}")


def read_edge_csv(path: Path) -> dict:
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            eid = row.get("edge_id", "")
            if eid:
                out[eid] = {
                    "speed_mps": float(row.get("speed_mps", 0) or 0),
                    "density_veh_per_m": float(row.get("density_veh_per_m", 0) or 0),
                    "flow_veh_per_s": float(row.get("flow_veh_per_s", 0) or 0),
                }
    return out


def load_truth(spec: str) -> dict:
    """spec이 .xml이면 SUMO edgedata(단위변환+sampled), 아니면 edge CSV."""
    p = Path(spec)
    if spec.endswith(".xml"):
        m = parse_sumo_edgedata_last_interval(p)
        return {e: v for e, v in m.items() if v.get("sampled_seconds", 0) > 0}
    return read_edge_csv(p)


def run_engine(net, route, vmax, ml, mn, sim_time, dt, out_csv) -> dict:
    cmd = [sys.executable, str(_HERE / "lane_cpu_simulator_mt.py"),
           "--net-file", net, "--route-file", route, "--model", "ctm",
           "--time-average", "--sim-time", str(sim_time), "--dt", str(dt),
           "--vmax-scale", str(vmax), "--major-left-factor", str(ml), "--minor-factor", str(mn),
           "--log-interval", "999999", "--output-csv", str(Path(out_csv).with_suffix(".lane.csv")),
           "--edge-output-csv", out_csv]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode != 0:
        raise SystemExit(f"engine 실패 vmax={vmax} ml={ml} mn={mn}")
    return read_edge_csv(Path(out_csv))


def score(engine_map: dict, truth: dict) -> dict:
    edges = sorted(set(engine_map) & set(truth))
    rf = np.array([truth[e]["flow_veh_per_s"] for e in edges]) * 3600.0
    tf = np.array([engine_map[e]["flow_veh_per_s"] for e in edges]) * 3600.0
    rs = np.array([truth[e]["speed_mps"] for e in edges])
    ts = np.array([engine_map[e]["speed_mps"] for e in edges])
    return {
        "n": len(edges),
        "flow_rmsn": M.rmsn(rf, tf),
        "flow_geh_lt5": M.geh_fraction(tf, rf, 5.0),
        "flow_spearman": M.spearman_rho(rf, tf),
        "speed_spearman": M.spearman_rho(rs, ts),
    }


def main():
    p = argparse.ArgumentParser(description="Held-out calibration for lane CTM")
    p.add_argument("--net-file", default="./map_import/gangnam4_generated.net.xml")
    p.add_argument("--train-route", required=True)
    p.add_argument("--train-truth", required=True, help="edge CSV 또는 SUMO edgedata .xml")
    p.add_argument("--train-sim-time", type=float, default=3600.0)
    p.add_argument("--test-route", required=True)
    p.add_argument("--test-truth", required=True)
    p.add_argument("--test-sim-time", type=float, default=3600.0)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--vmax-grid", default="0.3,0.4,0.5")
    p.add_argument("--ml-grid", default="0.5,0.7,1.0")
    p.add_argument("--mn-grid", default="0.3,0.5,1.0")
    p.add_argument("--tmp-dir", default="/tmp/calib")
    p.add_argument("--out-csv", default="/tmp/calib/calib_grid.csv")
    args = p.parse_args()

    tmp = Path(args.tmp_dir); tmp.mkdir(parents=True, exist_ok=True)
    train_truth = load_truth(args.train_truth)
    test_truth = load_truth(args.test_truth)
    log(f"train truth edges={len(train_truth)}, test truth edges={len(test_truth)}")

    grid = list(itertools.product(
        [float(x) for x in args.vmax_grid.split(",")],
        [float(x) for x in args.ml_grid.split(",")],
        [float(x) for x in args.mn_grid.split(",")],
    ))
    log(f"그리드 {len(grid)}개 조합 — train({Path(args.train_route).name})에서 flow RMSN 최소화")

    rows = []
    best = None
    for vmax, ml, mn in grid:
        em = run_engine(args.net_file, args.train_route, vmax, ml, mn,
                        args.train_sim_time, args.dt, str(tmp / f"tr_{vmax}_{ml}_{mn}.csv"))
        s = score(em, train_truth)
        rows.append({"vmax": vmax, "ml": ml, "mn": mn, **{f"train_{k}": v for k, v in s.items()}})
        log(f"  vmax={vmax} ml={ml} mn={mn}: flow_RMSN={s['flow_rmsn']:.4f} "
            f"GEH<5={s['flow_geh_lt5']*100:.1f}% flow_rho={s['flow_spearman']:.3f} "
            f"speed_rho={s['speed_spearman']:.3f}")
        # 목적함수: speed Spearman 최대화(혼잡 패턴 재현; scenario 간 비교 가능한 지표)
        if best is None or s["speed_spearman"] > best[1]["speed_spearman"]:
            best = ((vmax, ml, mn), s)

    (bv, bml, bmn), btrain = best
    log("=" * 60)
    log(f"최적(train, speed_rho 기준): vmax={bv} ml={bml} mn={bmn}  "
        f"speed_rho={btrain['speed_spearman']:.3f} flow_rho={btrain['flow_spearman']:.3f} "
        f"GEH<5={btrain['flow_geh_lt5']*100:.1f}%")

    # held-out 평가
    em_test = run_engine(args.net_file, args.test_route, bv, bml, bmn,
                         args.test_sim_time, args.dt, str(tmp / "test_best.csv"))
    stest = score(em_test, test_truth)
    log(f"held-out(test): speed_rho={stest['speed_spearman']:.3f} "
        f"flow_rho={stest['flow_spearman']:.3f} GEH<5={stest['flow_geh_lt5']*100:.1f}%")
    # scenario 간 비교 가능한 격차(순위상관 — 절대규모/RMSN과 달리 데모와 무관)
    log(f"train↔test 격차: speed_rho {btrain['speed_spearman']:.3f}→{stest['speed_spearman']:.3f} "
        f"(Δ{stest['speed_spearman']-btrain['speed_spearman']:+.3f}), "
        f"flow_rho {btrain['flow_spearman']:.3f}→{stest['flow_spearman']:.3f} "
        f"(Δ{stest['flow_spearman']-btrain['flow_spearman']:+.3f})")
    log("작을수록 일반화 잘 됨. (RMSN/GEH는 demand 규모에 의존하므로 gap 지표로는 순위상관 사용)")
    log("=" * 60)

    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        cols = ["vmax", "ml", "mn"] + [f"train_{k}" for k in btrain]
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    log(f"그리드 결과 저장: {out}")


if __name__ == "__main__":
    main()
