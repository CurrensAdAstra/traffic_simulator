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
import math
import xml.etree.ElementTree as ET
from pathlib import Path


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


def stats(xs):
    import statistics as st
    n = len(xs)
    if n == 0:
        return (0, 0, 0, 0)
    s = sorted(xs)
    mean = sum(xs) / n
    median = s[n // 2]
    p95 = s[min(n - 1, int(0.95 * n))]
    return (n, mean, median, p95)


def pearson(a, b):
    n = len(a)
    if n < 2:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    sab = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    saa = sum((x - ma) ** 2 for x in a)
    sbb = sum((y - mb) ** 2 for y in b)
    if saa <= 0 or sbb <= 0:
        return float("nan")
    return sab / math.sqrt(saa * sbb)


def main():
    p = argparse.ArgumentParser(description="Compare per-vehicle travel times: SUMO vs meso")
    p.add_argument("--sumo-tripinfo", required=True)
    p.add_argument("--meso-trips", required=True)
    p.add_argument("--total-vehicles", type=int, default=100000)
    args = p.parse_args()

    sumo = load_sumo_tripinfo(Path(args.sumo_tripinfo))
    meso = load_meso_trips(Path(args.meso_trips))
    log(f"SUMO 완주={len(sumo)}, meso 완주={len(meso)}, total={args.total_vehicles}")
    log(f"완주율: SUMO={100*len(sumo)/args.total_vehicles:.1f}%  meso={100*len(meso)/args.total_vehicles:.1f}%")

    sn, smean, smed, sp95 = stats(list(sumo.values()))
    mn, mmean, mmed, mp95 = stats(list(meso.values()))
    log(f"SUMO 통행시간: mean={smean:.1f}s median={smed:.1f}s p95={sp95:.1f}s (n={sn})")
    log(f"meso 통행시간: mean={mmean:.1f}s median={mmed:.1f}s p95={mp95:.1f}s (n={mn})")

    common = sorted(set(sumo) & set(meso))
    log(f"공통 완주 차량(id 매칭)={len(common)}")
    if common:
        sa = [sumo[v] for v in common]
        mb = [meso[v] for v in common]
        r = pearson(sa, mb)
        n = len(common)
        mae = sum(abs(x - y) for x, y in zip(sa, mb)) / n
        rmse = math.sqrt(sum((x - y) ** 2 for x, y in zip(sa, mb)) / n)
        bias = sum(y - x for x, y in zip(sa, mb)) / n  # meso - sumo
        log("=" * 52)
        log(f"차량별 통행시간 일치: Pearson r={r:.4f}  MAE={mae:.1f}s  RMSE={rmse:.1f}s")
        log(f"  bias(meso-SUMO) 평균={bias:+.1f}s")
        log("=" * 52)


if __name__ == "__main__":
    main()
