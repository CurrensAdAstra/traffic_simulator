#!/usr/bin/env python3
"""
합성 네트워크 시나리오 생성 — 외부 타당성(E3/E7 일반화)용 다중 네트워크.

인터넷 없이 재현 가능하도록 SUMO 표준 도구(netgenerate + randomTrips + duarouter)로
서로 다른 위상(topology)의 네트워크 + 라우팅된 수요를 만든다. 모두 Docker 컨테이너에서
실행. 결과는 map_import/synthetic/<name>/{net.xml, routes.rou.xml}.

토폴로지:
  grid   — 격자형(도심 블록)
  spider — 방사형(순환+방사 도로)
  random — 무작위(불규칙 도심)
각 net은 우리 엔진(meso/lane-CTM)과 SUMO 모두 로드 가능(2 lanes, 신호 포함).
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def log(m): print(f"[LOG] {m}")


def dock(args, inner: str):
    ws = Path(args.workspace).resolve()
    cmd = ["docker", "run", "--rm", "-v", f"{ws}:/workspace", "-w", "/workspace",
           args.docker_image, "bash", "-lc", inner]
    log("DOCKER: " + inner)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"docker 실행 실패: {inner[:80]}")


SCENARIOS = {
    # name: netgenerate flags (2 lanes, 신호 교차로, 적당한 속도)
    "grid":   "--grid --grid.number={n} --grid.length={L} -L 2 --tls.guess true --default.speed 13.9",
    "spider": "--spider --spider.arm-number={n} --spider.circle-number={n} --spider.space-radius={L} -L 2 --tls.guess true --default.speed 13.9",
    "random": "--rand --rand.iterations={ri} --rand.neighbor-dist 3 -L 2 --tls.guess true --default.speed 13.9 --seed {seed}",
}


def main():
    p = argparse.ArgumentParser(description="Generate synthetic SUMO scenarios for generalization tests")
    p.add_argument("--workspace", default="/home/mgkyung/ts")
    p.add_argument("--docker-image", default="gangnam4-cuda-sumo:latest")
    p.add_argument("--out-dir", default="map_import/synthetic")
    p.add_argument("--scenarios", default="grid,spider,random")
    p.add_argument("--grid-number", type=int, default=12)
    p.add_argument("--grid-length", type=int, default=200)
    p.add_argument("--spider-arms", type=int, default=8)
    p.add_argument("--rand-iterations", type=int, default=300)
    p.add_argument("--vehicles", type=int, default=20000)
    p.add_argument("--end", type=int, default=3600)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    for name in args.scenarios.split(","):
        d = f"{args.out_dir}/{name}"
        netf = f"{d}/net.xml"
        tripsf = f"{d}/trips.xml"
        routesf = f"{d}/routes.rou.xml"
        flags = SCENARIOS[name].format(n=args.grid_number if name != "spider" else args.spider_arms,
                                       L=args.grid_length, ri=args.rand_iterations, seed=args.seed)
        # 1) 네트워크 생성
        dock(args, f"mkdir -p /workspace/{d} && netgenerate {flags} -o /workspace/{netf}")
        # 2) 랜덤 수요(trips) — period로 차량수 제어: period = end/vehicles
        period = max(args.end / max(args.vehicles, 1), 0.01)
        dock(args, f"python3 $SUMO_HOME/tools/randomTrips.py -n /workspace/{netf} "
                   f"-e {args.end} -p {period:.4f} --seed {args.seed} "
                   f"-o /workspace/{tripsf} --validate")
        # 3) trips → routes (duarouter, depart 정렬)
        dock(args, f"duarouter -n /workspace/{netf} --route-files /workspace/{tripsf} "
                   f"-o /workspace/{routesf} --no-warnings true --ignore-errors true")
        # duarouter는 routesf.rou.xml로 쓰는 경우가 있어 확인
        dock(args, f"ls -la /workspace/{d}/ && grep -c '<vehicle' /workspace/{routesf} 2>/dev/null || true")
        log(f"[{name}] 생성 완료: {netf}, {routesf}")


if __name__ == "__main__":
    main()
