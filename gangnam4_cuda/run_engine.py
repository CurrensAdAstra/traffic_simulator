#!/usr/bin/env python3
"""
통합 엔진 디스패처 — 어떤 시뮬레이션 엔진을 쓸지 선택한다.

엔진(engine) × 백엔드(backend) 조합으로 4개 시뮬레이터 중 하나를 실행:

    engine=edge : 매크로스코픽, 도로(edge) 단위 (1세대)
    engine=lane : 차선(lane) 단위 + 회전 수요 차선변경 (2세대)
    backend=cpu : NumPy + ThreadPoolExecutor
    backend=gpu : CuPy RawKernel

나머지 인자는 선택된 시뮬레이터로 그대로 전달된다.

예)
    python3 gangnam4_cuda/run_engine.py --engine lane --backend cpu --steps 2000
    python3 gangnam4_cuda/run_engine.py --engine edge --backend gpu --dt 0.5
    python3 gangnam4_cuda/run_engine.py --list
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# (engine, backend) → 실행 스크립트 파일명 (이 파일과 같은 디렉터리 기준)
ENGINES: dict[tuple[str, str], str] = {
    ("edge", "cpu"): "cpu_edge_simulator_mt.py",
    ("edge", "gpu"): "cuda_edge_simulator.py",
    ("lane", "cpu"): "lane_cpu_simulator_mt.py",
    ("lane", "gpu"): "lane_cuda_simulator.py",
}

DESCRIPTIONS: dict[tuple[str, str], str] = {
    ("edge", "cpu"): "매크로스코픽 edge 단위 · CPU 멀티스레딩",
    ("edge", "gpu"): "매크로스코픽 edge 단위 · CUDA(1 edge=1 thread)",
    ("lane", "cpu"): "차선(lane) 단위 · CPU 멀티스레딩",
    ("lane", "gpu"): "차선(lane) 단위 · CUDA(1 lane=1 thread)",
}


def print_list() -> None:
    print("사용 가능한 엔진(engine × backend):")
    for (eng, be), script in ENGINES.items():
        print(f"  --engine {eng:4s} --backend {be:3s}  →  {script:28s}  ({DESCRIPTIONS[(eng, be)]})")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Traffic simulation engine dispatcher",
        epilog="알 수 없는 인자는 선택된 시뮬레이터로 그대로 전달됩니다.",
    )
    p.add_argument("--engine", choices=["edge", "lane"], help="시뮬레이션 엔진 선택")
    p.add_argument("--backend", choices=["cpu", "gpu"], help="연산 백엔드 선택")
    p.add_argument("--list", action="store_true", help="사용 가능한 엔진 목록 출력 후 종료")
    p.add_argument("--dry-run", action="store_true", help="실행할 명령만 출력하고 종료")
    args, passthrough = p.parse_known_args()

    if args.list:
        print_list()
        return

    if not args.engine or not args.backend:
        print("[ERROR] --engine 과 --backend 를 모두 지정해야 합니다.\n")
        print_list()
        raise SystemExit(2)

    script_name = ENGINES[(args.engine, args.backend)]
    script_path = Path(__file__).resolve().parent / script_name
    if not script_path.exists():
        print(f"[ERROR] 시뮬레이터 스크립트 없음: {script_path}")
        raise SystemExit(1)

    cmd = [sys.executable, str(script_path), *passthrough]
    print(f"[LOG] 선택: engine={args.engine}, backend={args.backend} → {script_name}")
    print(f"[LOG] 실행: {' '.join(cmd)}")

    if args.dry_run:
        return

    # 선택된 시뮬레이터로 프로세스 교체(인자 그대로 전달, cwd 유지)
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
