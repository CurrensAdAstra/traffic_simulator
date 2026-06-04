#!/usr/bin/env python3
"""
차량별 통행시간 비교: SUMO tripinfo vs mesoscopic 엔진 trip CSV.

메소스코픽 엔진의 신규 검증축 — 매크로(CTM)는 차량별 통행시간을 못 낸다.
두 소스에서 공통으로 완주한 차량을 id로 매칭하여:
 - 통행시간 분포(mean/median/p95) 비교
 - 차량별 Pearson r, MAE, RMSE
 - 완주율(arrival rate) 비교
"""

from __future__ import annotations

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # 표준 메트릭 공유


def log(m): print(f"[LOG] {m}")


def load_sumo_tripinfo(path: Path) -> dict[str, float]:
    """veh_id → duration(s). 부분 파일도 안전 파싱."""
    out: dict[str, float] = {}
    try:
        for _, el in ET.iterparse(path, events=("end",)):
            if el.tag == "tripinfo":
                vid = el.get("id", "")
                dur = el.get("duration")
                if vid and dur is not None:
                    out[vid] = float(dur)
                el.clear()
    except ET.ParseError:
        # 부분(미완료) XML — 지금까지 읽은 것만 사용
        log(f"tripinfo 부분 파싱(미완료 파일): {len(out)}건 확보")
    return out


def load_meso_trips(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            out[row["veh_id"]] = float(row["travel_time_s"])
    return out


def dist_stats(xs: np.ndarray):
    if xs.size == 0:
        return (0, 0.0, 0.0, 0.0)
    return (int(xs.size), float(xs.mean()), float(np.median(xs)), float(np.percentile(xs, 95)))


def main():
    p = argparse.ArgumentParser(description="Compare per-vehicle travel times: SUMO vs meso")
    p.add_argument("--sumo-tripinfo", required=True)
    p.add_argument("--meso-trips", required=True)
    p.add_argument("--total-vehicles", type=int, default=100000)
    args = p.parse_args()

    sumo = load_sumo_tripinfo(Path(args.sumo_tripinfo))
    meso = load_meso_trips(Path(args.meso_trips))
    log(f"SUMO 완주={len(sumo)}, meso 완주={len(meso)}, total={args.total_vehicles}")
    s_rate = 100 * len(sumo) / args.total_vehicles
    m_rate = 100 * len(meso) / args.total_vehicles
    log(f"완주율: SUMO={s_rate:.1f}%  meso={m_rate:.1f}%  (차이 {abs(s_rate-m_rate):.1f}%p)")

    sa_all = np.array(list(sumo.values()), float)
    ma_all = np.array(list(meso.values()), float)
    sn, smean, smed, sp95 = dist_stats(sa_all)
    mn, mmean, mmed, mp95 = dist_stats(ma_all)
    log(f"SUMO 통행시간: mean={smean:.1f}s median={smed:.1f}s p95={sp95:.1f}s (n={sn})")
    log(f"meso 통행시간: mean={mmean:.1f}s median={mmed:.1f}s p95={mp95:.1f}s (n={mn})")
    # 분포(완주 차량 전체) KS — id 매칭 없이 분포 형태 비교
    log(f"통행시간 분포 KS(전체) = {M.ks_statistic(sa_all, ma_all):.4f}")

    common = sorted(set(sumo) & set(meso))
    log(f"공통 완주 차량(id 매칭)={len(common)}")
    if common:
        sa = np.array([sumo[v] for v in common], float)
        mb = np.array([meso[v] for v in common], float)
        log("=" * 60)
        log(f"차량별 통행시간 일치(n={len(common)}):")
        log(f"  Pearson r={M.pearson_r(sa, mb):.4f}  Spearman rho={M.spearman_rho(sa, mb):.4f}")
        log(f"  MAPE={M.mape(sa, mb):.1f}%  bias(meso-SUMO)={M.bias(sa, mb):+.1f}s")
        log(f"  RMSN={M.rmsn(sa, mb):.4f}  KS(matched)={M.ks_statistic(sa, mb):.4f}")
        log("=" * 60)


if __name__ == "__main__":
    main()
