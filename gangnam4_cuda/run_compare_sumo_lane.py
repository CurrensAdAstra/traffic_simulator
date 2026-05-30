#!/usr/bin/env python3
"""
SUMO(미시 기준) vs 차선(lane) 엔진 검증 리포트.

목적: 비전문가도 방어 가능한 "정량 검증"을 만든다.
 - lane 엔진을 edge 단위로 집계한 뒤, SUMO edgedata(마지막 interval)와 edge_id로 매칭
 - 속도/밀도/유량에 대해 상관계수(Pearson r), MAE, RMSE 계산
 - 혼잡 상위(worst-K) edge 집합의 일치도(Jaccard) — "혼잡 hotspot을 재현하는가"

SUMO를 직접 실행하거나(--run-sumo), 미리 만든 edgedata xml(--sumo-edgedata)을 쓴다.
SUMO가 없는 환경에서는 --ref-edge-csv 로 임의의 edge 기준 CSV(예: edge 엔진 출력)와
비교하여 하니스 자체를 검증할 수 있다(엔진 간 일관성 체크).

예)
  # 실제 검증(SUMO edgedata 보유 시)
  python3 gangnam4_cuda/run_compare_sumo_lane.py \
      --net-file map_import/gangnam4_generated.net.xml \
      --route-file map_import/gangnam4_generated.gpu_compatible.rou.xml \
      --sumo-edgedata map_import/edgedata.xml --steps 2000

  # SUMO 없이 엔진 간 일관성 체크
  python3 gangnam4_cuda/run_compare_sumo_lane.py \
      --net-file ... --route-file ... \
      --ref-edge-csv gangnam4_cuda/results/edge_state_cpu_mt.csv --steps 200
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


def run_lane_engine(args, lane_edge_csv: Path) -> None:
    """lane CPU 엔진을 실행해 edge 집계 CSV를 생성."""
    lane_csv = lane_edge_csv.with_suffix(".lane.csv")
    # SUMO 검증 기본: CTM 모델 + 시간평균 + SUMO와 같은 sim_time
    cmd = [
        sys.executable,
        str(_HERE / "lane_cpu_simulator_mt.py"),
        "--net-file", str(args.net_file),
        "--steps", str(args.steps),
        "--dt", str(args.dt),
        "--lane-change-rate", str(args.lane_change_rate),
        "--sim-duration", str(args.duration_sec),
        "--model", args.model,
        "--sim-time", str(args.engine_sim_time if args.engine_sim_time > 0 else args.duration_sec),
        "--output-csv", str(lane_csv),
        "--edge-output-csv", str(lane_edge_csv),
    ]
    if args.time_average:
        cmd.append("--time-average")
    if args.route_file:
        cmd += ["--route-file", str(args.route_file)]
    log("CMD: " + " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        fail(f"lane 엔진 실행 실패(rc={rc})")


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


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def _err_stats(ref: list[float], test: list[float]) -> tuple[float, float, float]:
    """(Pearson r, MAE, RMSE)."""
    n = len(ref)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mae = sum(abs(a - b) for a, b in zip(ref, test)) / n
    rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(ref, test)) / n)
    return _pearson(ref, test), mae, rmse


def compare(ref: EdgeMetrics, test: EdgeMetrics, topk: int, report_csv: Path) -> dict:
    edges = sorted(set(ref) & set(test))
    if not edges:
        fail("매칭되는 edge가 0개 — net/route/edge_id 정합성 확인 필요")

    metrics = ["speed_mps", "density_veh_per_m", "flow_veh_per_s"]
    summary: dict[str, dict[str, float]] = {}
    for m in metrics:
        r_vals = [ref[e][m] for e in edges]
        t_vals = [test[e][m] for e in edges]
        r, mae, rmse = _err_stats(r_vals, t_vals)
        summary[m] = {"pearson_r": r, "mae": mae, "rmse": rmse}

    # 혼잡 hotspot 일치도: 속도 하위 K edge 집합의 Jaccard
    k = min(topk, len(edges))
    ref_worst = set(sorted(edges, key=lambda e: ref[e]["speed_mps"])[:k])
    test_worst = set(sorted(edges, key=lambda e: test[e]["speed_mps"])[:k])
    inter = len(ref_worst & test_worst)
    union = len(ref_worst | test_worst)
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
    p.add_argument("--out-prefix", default="./gangnam4_cuda/results/compare_lane")
    # lane 엔진 모델 옵션(SUMO와 공정 비교를 위한 기본값)
    p.add_argument("--model", default="ctm", choices=["lwr", "ctm"],
                   help="lane 엔진 갱신 모델(기본 ctm: spillback 포함)")
    p.add_argument("--time-average", action="store_true", default=True,
                   help="lane 엔진 결과를 시간평균하여 SUMO edgeData와 정렬(기본 ON)")
    p.add_argument("--no-time-average", action="store_false", dest="time_average")
    p.add_argument("--engine-sim-time", type=float, default=0.0,
                   help="lane 엔진의 모델 시뮬레이션 시간(초). 0이면 --duration-sec와 동일하게 정렬")
    # 기준(reference) 선택: 셋 중 하나
    p.add_argument("--sumo-edgedata", default=None, help="기존 SUMO edgedata xml 경로")
    p.add_argument("--run-sumo", action="store_true", help="SUMO를 직접 실행해 edgedata 생성")
    p.add_argument("--sumo", default="sumo", help="SUMO 실행 바이너리")
    p.add_argument("--ref-edge-csv", default=None, help="SUMO 대신 비교할 edge 기준 CSV(엔진 간 일관성 체크)")
    # 이미 만든 lane edge 집계 CSV 재사용(엔진 재실행 생략)
    p.add_argument("--lane-edge-csv", default=None, help="기존 lane edge 집계 CSV 재사용")
    args = p.parse_args()

    out_prefix = Path(f"{args.out_prefix}_{time.strftime('%Y%m%d_%H%M%S')}")

    # 1) lane 엔진 결과(edge 집계) 확보
    if args.lane_edge_csv:
        lane_edge_csv = Path(args.lane_edge_csv)
        if not lane_edge_csv.exists():
            fail(f"lane-edge-csv 없음: {lane_edge_csv}")
        log(f"기존 lane edge 집계 재사용: {lane_edge_csv}")
    else:
        lane_edge_csv = Path(f"{out_prefix}_lane_edge.csv")
        run_lane_engine(args, lane_edge_csv)
    lane_metrics = read_edge_metric_csv(lane_edge_csv)
    log(f"lane edge 지표: {len(lane_metrics)}개 edge")

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
    res = compare(ref_metrics, lane_metrics, args.topk, report_csv)

    # 4) 요약 출력 + 저장
    log("=" * 56)
    log(f"검증 리포트: 기준={ref_name}  vs  lane 엔진")
    log(f"  매칭 edge 수: {res['matched_edges']}")
    for m, s in res["metrics"].items():
        log(f"  {m:20s}  r={s['pearson_r']:.4f}  MAE={s['mae']:.4f}  RMSE={s['rmse']:.4f}")
    log(f"  혼잡 hotspot(top-{res['hotspot_topk']}) 일치: "
        f"{res['hotspot_overlap']}  Jaccard={res['hotspot_jaccard']:.4f}")
    log(f"  edgewise CSV: {report_csv}")
    log("=" * 56)

    summary_csv = Path(f"{out_prefix}_summary.csv")
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "pearson_r", "mae", "rmse"])
        for m, s in res["metrics"].items():
            w.writerow([m, f"{s['pearson_r']:.6f}", f"{s['mae']:.6f}", f"{s['rmse']:.6f}"])
        w.writerow(["hotspot_jaccard", f"{res['hotspot_jaccard']:.6f}", res["hotspot_overlap"], f"top{res['hotspot_topk']}"])
    log(f"요약 저장: {summary_csv}")


if __name__ == "__main__":
    main()
