#!/usr/bin/env python3
"""
SUMO 대비 엔진 검증 하니스 — 어떤 엔진이든 같은 데이터로 비교한다.

핵심 약속: 같은 입력(`--net-file`, `--route-file`, `--duration-sec`)에 대해
`--engine {edge,lane} --backend {cpu,gpu}` 만 바꾸면 **동일한 검증 리포트**가
나온다. 모든 엔진의 edge-keyed CSV 출력에서 `(edge_id, speed_mps,
density_veh_per_m, flow_veh_per_s)` 컬럼을 읽어 동일하게 채점.

리포트:
 - 속도/밀도/유량 Pearson r, MAE, RMSE
 - 혼잡 상위(worst-K) edge 집합의 Jaccard — "혼잡 hotspot을 재현하는가"

기준(reference):
 - `--run-sumo`           SUMO를 직접 실행
 - `--sumo-edgedata FILE` 기존 SUMO edgedata xml 사용
 - `--ref-edge-csv FILE`  SUMO 대신 임의의 edge 기준 CSV(엔진 간 일관성 체크용)

예)
  # 실 검증(SUMO 있을 때) — lane CTM 엔진
  python3 gangnam4_cuda/run_compare_sumo.py --run-sumo \
      --engine lane --backend cpu \
      --net-file map_import/gangnam4_generated.net.xml \
      --route-file map_import/gangnam4_generated.gpu_compatible.rou.xml \
      --duration-sec 3600

  # 같은 데이터, edge 엔진으로 비교
  python3 gangnam4_cuda/run_compare_sumo.py --run-sumo \
      --engine edge --backend cpu --net-file ... --route-file ... --duration-sec 3600

  # SUMO 없이 엔진 간 일관성 체크
  python3 gangnam4_cuda/run_compare_sumo.py --ref-edge-csv edge_state.csv \
      --engine lane --backend cpu --net-file ... --route-file ...
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict

# 기존 비교 스크립트의 SUMO edgedata 파서/유틸 재사용
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from run_compare_sumo_cpu_gpu import (  # type: ignore
    parse_sumo_edgedata_last_interval,
    to_cfg_relative,
)


def log(msg: str) -> None:
    print(f"[LOG] {msg}")


def fail(msg: str) -> None:
    print(f"[ERROR] {msg}")
    raise SystemExit(1)


EdgeMetrics = Dict[str, Dict[str, float]]


def read_edge_metric_csv(path: Path) -> EdgeMetrics:
    """edge_id 키 CSV에서 speed/density/flow를 읽는다(컬럼 이름은 두 엔진 공통)."""
    out: EdgeMetrics = {}
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            eid = row.get("edge_id", "")
            if not eid:
                continue
            out[eid] = {
                "speed_mps": float(row.get("speed_mps", "0") or 0.0),
                "density_veh_per_m": float(row.get("density_veh_per_m", "0") or 0.0),
                "flow_veh_per_s": float(row.get("flow_veh_per_s", "0") or 0.0),
            }
    return out


def run_engine_subprocess(args, edge_csv: Path) -> None:
    """선택된 엔진(--engine/--backend)을 run_engine.py 디스패처로 실행해
    edge_id-keyed CSV를 생성. edge 엔진은 출력 CSV 자체가 edge-CSV이고,
    lane 엔진은 별도 --edge-output-csv 인자를 통해 edge 집계 CSV를 만든다.
    이 함수는 어느 쪽이든 같은 `edge_csv` 경로에 결과를 남긴다.
    """
    cmd = [
        sys.executable, str(_HERE / "run_engine.py"),
        "--engine", args.engine,
        "--backend", args.backend,
        # 공통 인자
        "--net-file", str(args.net_file),
        "--steps", str(args.steps),
        "--dt", str(args.dt),
        "--sim-duration", str(args.duration_sec),
    ]
    if args.route_file:
        cmd += ["--route-file", str(args.route_file)]

    if args.engine == "lane":
        # lane 엔진: edge 집계 CSV가 비교 대상. --output-csv는 per-lane 부산물.
        lane_csv = edge_csv.with_suffix(".lane.csv")
        cmd += [
            "--output-csv", str(lane_csv),
            "--edge-output-csv", str(edge_csv),
            "--lane-change-rate", str(args.lane_change_rate),
            "--model", args.model,
            "--sim-time", str(args.engine_sim_time if args.engine_sim_time > 0 else args.duration_sec),
            "--vmax-scale", str(args.vmax_scale),
            "--junction-cap-factor", str(args.junction_cap_factor),
            "--major-left-factor", str(args.major_left_factor),
            "--minor-factor", str(args.minor_factor),
        ]
        if args.time_average:
            cmd.append("--time-average")
    else:
        # edge 엔진: --output-csv가 곧 edge-CSV.
        cmd += ["--output-csv", str(edge_csv)]

    log("CMD: " + " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        fail(f"엔진({args.engine}/{args.backend}) 실행 실패(rc={rc})")


def run_sumo_edgedata(args, out_prefix: Path) -> Path:
    """SUMO를 실행해 edgedata xml을 생성하고 경로를 반환."""
    cfg = out_prefix.with_suffix(".sumocfg")
    edgedata = Path(f"{out_prefix}_sumo_edge.xml")
    add_file = Path(f"{out_prefix}_edgedata.add.xml")
    sumo_log = Path(f"{out_prefix}_sumo.log")

    net_rel = to_cfg_relative(Path(args.net_file), cfg)
    route_rel = to_cfg_relative(Path(args.route_file), cfg)
    edge_rel = to_cfg_relative(edgedata, cfg)
    add_rel = to_cfg_relative(add_file, cfg)

    # edgeData 수집기를 additional 파일로 정의(전체 기간 1 interval)
    add_file.write_text(
        "\n".join([
            "<additional>",
            f'  <edgeData id="ed" file="{edge_rel}" begin="0" end="{int(args.duration_sec)}"/>',
            "</additional>",
        ]),
        encoding="utf-8",
    )
    cfg.write_text(
        "\n".join([
            "<configuration>",
            "  <input>",
            f'    <net-file value="{net_rel}"/>',
            f'    <route-files value="{route_rel}"/>',
            f'    <additional-files value="{add_rel}"/>',
            "  </input>",
            "  <time>",
            '    <begin value="0"/>',
            f'    <end value="{int(args.duration_sec)}"/>',
            "  </time>",
            "</configuration>",
        ]),
        encoding="utf-8",
    )
    cmd = [args.sumo, "-c", str(cfg), "--no-warnings", "true", "--log", str(sumo_log)]
    log("CMD: " + " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        fail(f"SUMO 실행 실패(rc={rc})")
    if not edgedata.exists():
        fail(f"SUMO edgedata 미생성: {edgedata}")
    return edgedata


import numpy as _np
import metrics as _M  # 표준 교통 메트릭(GEH/RMSN/Spearman/KS/precision-recall)


def compare(ref: EdgeMetrics, test: EdgeMetrics, topk: int, report_csv: Path) -> dict:
    edges = sorted(set(ref) & set(test))
    if not edges:
        fail("매칭되는 edge가 0개 — net/route/edge_id 정합성 확인 필요")

    metrics_keys = ["speed_mps", "density_veh_per_m", "flow_veh_per_s"]
    summary: dict[str, dict[str, float]] = {}
    arrs: dict[str, tuple] = {}
    for m in metrics_keys:
        r_vals = _np.array([ref[e][m] for e in edges], float)
        t_vals = _np.array([test[e][m] for e in edges], float)
        arrs[m] = (r_vals, t_vals)
        summary[m] = {
            "pearson_r": _M.pearson_r(r_vals, t_vals),
            "spearman_rho": _M.spearman_rho(r_vals, t_vals),
            "rmsn": _M.rmsn(r_vals, t_vals),
            "mae": float(_np.mean(_np.abs(t_vals - r_vals))),
            "rmse": float(_np.sqrt(_np.mean((t_vals - r_vals) ** 2))),
        }

    # flow GEH (교통 표준): veh/s → veh/h 환산 후 GEH<5 비율
    rf, tf = arrs["flow_veh_per_s"]
    summary["flow_veh_per_s"]["geh_lt5_frac"] = _M.geh_fraction(tf * 3600.0, rf * 3600.0, 5.0)

    # 혼잡 hotspot: 속도 하위 K — Jaccard(하위호환) + precision/recall@K + Spearman(전체)
    k = min(topk, len(edges))
    rs, ts = arrs["speed_mps"]
    prec, rec = _M.precision_recall_at_k(rs, ts, k, lowest=True)
    ref_worst = set(_np.argsort(rs)[:k]); test_worst = set(_np.argsort(ts)[:k])
    inter = len(ref_worst & test_worst); union = len(ref_worst | test_worst)
    jaccard = inter / union if union else float("nan")

    # edgewise 리포트 저장
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    with report_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "edge_id",
            "ref_speed", "lane_speed", "abs_err_speed",
            "ref_density", "lane_density", "abs_err_density",
            "ref_flow", "lane_flow", "abs_err_flow",
        ])
        for e in edges:
            rr, tt = ref[e], test[e]
            w.writerow([
                e,
                f"{rr['speed_mps']:.6f}", f"{tt['speed_mps']:.6f}", f"{abs(rr['speed_mps']-tt['speed_mps']):.6f}",
                f"{rr['density_veh_per_m']:.6f}", f"{tt['density_veh_per_m']:.6f}", f"{abs(rr['density_veh_per_m']-tt['density_veh_per_m']):.6f}",
                f"{rr['flow_veh_per_s']:.6f}", f"{tt['flow_veh_per_s']:.6f}", f"{abs(rr['flow_veh_per_s']-tt['flow_veh_per_s']):.6f}",
            ])

    return {
        "matched_edges": len(edges),
        "metrics": summary,
        "hotspot_topk": k,
        "hotspot_jaccard": jaccard,
        "hotspot_overlap": f"{inter}/{k}",
        "hotspot_precision": prec,
        "hotspot_recall": rec,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="SUMO vs lane-engine validation report")
    p.add_argument("--net-file", default="./map_import/gangnam4_generated.net.xml")
    p.add_argument("--route-file", default="./map_import/gangnam4_generated.gpu_compatible.rou.xml")
    p.add_argument("--duration-sec", type=float, default=86400.0)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--lane-change-rate", type=float, default=0.5)
    p.add_argument("--topk", type=int, default=50, help="혼잡 hotspot 일치도 비교 edge 수")
    p.add_argument("--out-prefix", default="./gangnam4_cuda/results/compare")
    # 엔진 선택 — 같은 데이터로 엔진만 바꿔 비교할 수 있도록 일반화
    p.add_argument("--engine", default="lane", choices=["edge", "lane"],
                   help="시뮬레이션 엔진(기본 lane)")
    p.add_argument("--backend", default="cpu", choices=["cpu", "gpu"],
                   help="연산 백엔드(기본 cpu)")
    # lane 엔진 전용 옵션(--engine edge에서는 무시)
    p.add_argument("--model", default="ctm", choices=["lwr", "ctm"],
                   help="lane 엔진 갱신 모델(기본 ctm: spillback 포함)")
    p.add_argument("--time-average", action="store_true", default=True,
                   help="lane 엔진 결과를 시간평균하여 SUMO edgeData와 정렬(기본 ON)")
    p.add_argument("--no-time-average", action="store_false", dest="time_average")
    p.add_argument("--engine-sim-time", type=float, default=0.0,
                   help="lane 엔진의 모델 시뮬레이션 시간(초). 0이면 --duration-sec와 동일하게 정렬")
    p.add_argument("--vmax-scale", type=float, default=1.0,
                   help="FD 보정 — 모든 lane의 vmax에 곱함(기본 1.0 = 보정 없음)")
    p.add_argument("--junction-cap-factor", type=float, default=1e9,
                   help="교차로 처리용량 계수(기본 1e9=제약 없음)")
    p.add_argument("--major-left-factor", type=float, default=1.0,
                   help="HCM Rank 2: major 좌/U-turn 용량 계수")
    p.add_argument("--minor-factor", type=float, default=1.0,
                   help="HCM Rank 3-4: minor 모든 movement 용량 계수")
    # 기준(reference) 선택: 셋 중 하나
    p.add_argument("--sumo-edgedata", default=None, help="기존 SUMO edgedata xml 경로")
    p.add_argument("--run-sumo", action="store_true", help="SUMO를 직접 실행해 edgedata 생성")
    p.add_argument("--sumo", default="sumo", help="SUMO 실행 바이너리")
    p.add_argument("--ref-edge-csv", default=None, help="SUMO 대신 비교할 edge 기준 CSV(엔진 간 일관성 체크)")
    # 이미 만든 엔진 출력 CSV 재사용(엔진 재실행 생략)
    p.add_argument("--engine-edge-csv", default=None, help="기존 엔진 edge-keyed CSV 재사용")
    args = p.parse_args()

    out_prefix = Path(f"{args.out_prefix}_{args.engine}_{args.backend}_{time.strftime('%Y%m%d_%H%M%S')}")

    # 1) 엔진 결과(edge-keyed CSV) 확보
    if args.engine_edge_csv:
        engine_edge_csv = Path(args.engine_edge_csv)
        if not engine_edge_csv.exists():
            fail(f"engine-edge-csv 없음: {engine_edge_csv}")
        log(f"기존 엔진 출력 재사용: {engine_edge_csv}")
    else:
        engine_edge_csv = Path(f"{out_prefix}_edge.csv")
        run_engine_subprocess(args, engine_edge_csv)
    engine_metrics = read_edge_metric_csv(engine_edge_csv)
    log(f"엔진({args.engine}/{args.backend}) edge 지표: {len(engine_metrics)}개 edge")

    # 2) 기준(reference) 지표 확보
    if args.ref_edge_csv:
        ref_metrics = read_edge_metric_csv(Path(args.ref_edge_csv))
        ref_name = f"REF_CSV({Path(args.ref_edge_csv).name})"
    else:
        if args.sumo_edgedata:
            edgedata = Path(args.sumo_edgedata)
            if not edgedata.exists():
                fail(f"sumo-edgedata 없음: {edgedata}")
        elif args.run_sumo:
            edgedata = run_sumo_edgedata(args, out_prefix)
        else:
            fail("기준 미지정: --sumo-edgedata / --run-sumo / --ref-edge-csv 중 하나 필요")
        # 단위 정규화(veh/km→veh/m, veh/h→veh/s)는 parse 함수가 처리. 여기선 관측 edge만 추림.
        sumo_map = parse_sumo_edgedata_last_interval(edgedata)
        ref_metrics = {}
        skipped = 0
        for eid, v in sumo_map.items():
            if v.get("sampled_seconds", 0.0) <= 0.0:
                skipped += 1
                continue
            ref_metrics[eid] = {
                "speed_mps": v.get("speed_mps", 0.0),
                "density_veh_per_m": v.get("density_veh_per_m", 0.0),
                "flow_veh_per_s": v.get("flow_veh_per_s", 0.0),
            }
        ref_name = "SUMO"
        log(f"SUMO 관측 edge={len(ref_metrics)} (미관측 {skipped}개 제외)")
    log(f"기준({ref_name}) edge 지표: {len(ref_metrics)}개 edge")

    # 3) 비교
    report_csv = Path(f"{out_prefix}_edgewise.csv")
    res = compare(ref_metrics, engine_metrics, args.topk, report_csv)

    # 4) 요약 출력 + 저장
    log("=" * 64)
    log(f"검증 리포트: 기준={ref_name}  vs  엔진({args.engine}/{args.backend}, model={args.model if args.engine=='lane' else 'n/a'})")
    log(f"  매칭 edge 수: {res['matched_edges']}")
    for m, s in res["metrics"].items():
        extra = f"  GEH<5={s['geh_lt5_frac']*100:.1f}%" if "geh_lt5_frac" in s else ""
        log(f"  {m:20s}  r={s['pearson_r']:.4f}  rho={s['spearman_rho']:.4f}  "
            f"RMSN={s['rmsn']:.4f}  MAE={s['mae']:.4f}{extra}")
    log(f"  혼잡 hotspot(top-{res['hotspot_topk']}): "
        f"precision={res['hotspot_precision']:.3f} recall={res['hotspot_recall']:.3f} "
        f"(Jaccard={res['hotspot_jaccard']:.3f}, overlap {res['hotspot_overlap']})")
    log(f"  edgewise CSV: {report_csv}")
    log("=" * 64)

    summary_csv = Path(f"{out_prefix}_summary.csv")
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "pearson_r", "spearman_rho", "rmsn", "mae", "rmse", "geh_lt5_frac"])
        for m, s in res["metrics"].items():
            w.writerow([m, f"{s['pearson_r']:.6f}", f"{s['spearman_rho']:.6f}",
                        f"{s['rmsn']:.6f}", f"{s['mae']:.6f}", f"{s['rmse']:.6f}",
                        f"{s.get('geh_lt5_frac', float('nan')):.6f}"])
        w.writerow(["hotspot_topk", res["hotspot_topk"], "", "", "", "", ""])
        w.writerow(["hotspot_precision", f"{res['hotspot_precision']:.6f}", "", "", "", "", ""])
        w.writerow(["hotspot_recall", f"{res['hotspot_recall']:.6f}", "", "", "", "", ""])
        w.writerow(["hotspot_jaccard", f"{res['hotspot_jaccard']:.6f}", res["hotspot_overlap"], "", "", "", ""])
    log(f"요약 저장: {summary_csv}")


if __name__ == "__main__":
    main()
