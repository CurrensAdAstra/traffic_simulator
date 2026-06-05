#!/usr/bin/env python3
"""
수요(route) 파일을 목표 차량 수로 스케일 — 스케일링 실험(E2)용.

- 축소: 출발순 앞쪽 N대만(결정론적).
- 확대: 기존 route id를 재사용(차량은 route 공유 가능)하여 차량 수를 늘린다.
  추가 차량의 depart는 기존 depart 분포에서 복원추출 + 소량 지터(±0.5*간격).
출력은 depart 정렬된 .rou.xml (SUMO/meso 모두 사용 가능).
난수는 numpy default_rng(seed)로 결정론적.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description="Scale a SUMO route file to a target vehicle count")
    p.add_argument("--in-route", required=True)
    p.add_argument("--out-route", required=True)
    p.add_argument("--target", type=int, required=True, help="목표 차량 수")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    routes: dict[str, str] = {}
    veh: list[tuple[str, float]] = []  # (route_id, depart)
    for _, el in ET.iterparse(args.in_route, events=("end",)):
        if el.tag == "route":
            rid = el.get("id")
            if rid:
                routes[rid] = el.get("edges", "")
            el.clear()
        elif el.tag == "vehicle":
            rid = el.get("route", "")
            if rid in routes or rid == "":
                veh.append((rid, float(el.get("depart", "0"))))
            # vehicle 내부 route 처리
            rn = el.find("route")
            if rn is not None and rid == "":
                k = f"_inl{len(routes)}"
                routes[k] = rn.get("edges", "")
                veh[-1] = (k, veh[-1][1])
            el.clear()

    n0 = len(veh)
    departs = np.array([d for _, d in veh], float)
    rids = [r for r, _ in veh]
    T = float(departs.max()) if n0 else 3600.0

    if args.target <= n0:
        # 축소: depart 정렬 후 앞쪽 target
        order = np.argsort(departs, kind="stable")[:args.target]
        sel = [(rids[i], departs[i]) for i in order]
    else:
        # 확대: 원본 전체 + (target-n0)개 복제(route 재사용, depart 복원추출+지터)
        sel = list(zip(rids, departs.tolist()))
        extra = args.target - n0
        idx = rng.integers(0, n0, size=extra)
        jitter = rng.uniform(-0.5, 0.5, size=extra)
        for k in range(extra):
            i = int(idx[k])
            d = float(min(max(departs[i] + jitter[k], 0.0), T))
            sel.append((rids[i], d))

    sel.sort(key=lambda x: x[1])
    used_routes = {r for r, _ in sel if r in routes}

    out = Path(args.out_route)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("<routes>\n")
        for rid in sorted(used_routes):
            f.write(f'  <route id="{rid}" edges="{routes[rid]}"/>\n')
        for k, (rid, d) in enumerate(sel):
            f.write(f'  <vehicle id="s{k}" depart="{d:.2f}" route="{rid}"/>\n')
        f.write("</routes>\n")
    print(f"[LOG] scaled {n0} -> {len(sel)} vehicles ({len(used_routes)} routes) -> {out}")


if __name__ == "__main__":
    main()
