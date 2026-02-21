#!/usr/bin/env python3
"""
강남4구 SUMO 시뮬레이션 + CUDA 후처리 벤치마크 스크립트.

핵심 개념
- SUMO 자체는 CPU 기반 시뮬레이터이므로 CUDA는 "결과 후처리" 단계에서 사용.
- tripinfo XML을 스트리밍 파싱하고, GPU(CUDA)에서 통계 계산 성능을 측정.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class GpuBackend:
    name: str
    available: bool
    reason: str


def log(msg: str) -> None:
    print(f"[LOG] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def fail(msg: str, code: int = 1) -> None:
    print(f"[ERROR] {msg}")
    raise SystemExit(code)


def is_xml_completed(path: Path, closing_tag: bytes = b"</tripinfos>") -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb") as f:
        try:
            f.seek(max(path.stat().st_size - 8192, 0))
        except OSError:
            return False
        tail = f.read()
    return closing_tag in tail


def wait_for_tripinfo_completion(tripinfo_path: Path, timeout_sec: int = 7200, poll_sec: float = 2.0) -> None:
    """Race condition 방지: tripinfo XML 종료 태그가 기록될 때까지 대기."""
    log(f"tripinfo 완료 대기 시작: {tripinfo_path} (timeout={timeout_sec}s)")
    t0 = time.perf_counter()
    last_size = -1

    while True:
        if is_xml_completed(tripinfo_path):
            log("tripinfo XML 완료 확인(종료 태그 감지)")
            return

        elapsed = time.perf_counter() - t0
        if elapsed >= timeout_sec:
            fail("tripinfo XML 완료 대기 타임아웃(종료 태그 미감지)")

        try:
            size = tripinfo_path.stat().st_size if tripinfo_path.exists() else 0
        except OSError:
            size = 0

        if size != last_size:
            log(f"tripinfo 작성중... size={size} bytes, elapsed={elapsed:.1f}s")
            last_size = size

        time.sleep(poll_sec)


def create_runtime_sumocfg(base_cfg: Path, runtime_cfg: Path, summary: Path, tripinfo: Path, stat_output: Path) -> None:
    """기존 sumocfg를 복사하면서 output 파일 경로를 실행별 고유 파일로 치환."""
    tree = ET.parse(base_cfg)
    root = tree.getroot()

    output = root.find("output")
    if output is None:
        output = ET.SubElement(root, "output")

    def set_output(tag: str, value: str) -> None:
        node = output.find(tag)
        if node is None:
            node = ET.SubElement(output, tag)
        node.set("value", value)

    cfg_dir = runtime_cfg.parent

    def rel_to_cfg_dir(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(cfg_dir.resolve()))
        except Exception:
            return path.name

    set_output("summary-output", rel_to_cfg_dir(summary))
    set_output("tripinfo-output", rel_to_cfg_dir(tripinfo))
    set_output("statistic-output", rel_to_cfg_dir(stat_output))

    runtime_cfg.parent.mkdir(parents=True, exist_ok=True)
    tree.write(runtime_cfg, encoding="UTF-8", xml_declaration=True)
    log(f"runtime sumocfg 생성: {runtime_cfg}")


def check_paths(cfg: Path, tripinfo: Path, summary: Path) -> None:
    log("입력 경로 점검 시작")
    checks = [cfg]
    for p in checks:
        if not p.exists():
            fail(f"필수 파일 없음: {p}")
        log(f"존재 확인: {p} (size={p.stat().st_size} bytes)")

    log("진단 후보(6개) 출력")
    candidates = [
        "SUMO 바이너리 미설치 또는 PATH 미설정",
        "sumocfg 내부 net/route 상대경로 불일치",
        "route 정렬/유효성 이슈로 시뮬레이션 성능 저하",
        "CUDA 런타임 또는 드라이버 불일치",
        "GPU 메모리 부족(24GB 초과 입력)",
        "tripinfo XML 파싱 병목(대용량 I/O)",
    ]
    for i, c in enumerate(candidates, 1):
        log(f"  후보{i}: {c}")

    log("가장 가능성 높은 원인(2개) 가정")
    log("  가정A: sumocfg의 상대경로 기준(실행 cwd)이 맞지 않아 입력 파일 로딩 실패")
    log("  가정B: CUDA 백엔드(cupy/torch) 미설치 또는 cuda 미활성")


def run_cmd(cmd: List[str], cwd: Path | None = None, env: Dict[str, str] | None = None) -> Tuple[int, float]:
    t0 = time.perf_counter()
    log(f"CMD: {' '.join(shlex.quote(x) for x in cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env)
    dt = time.perf_counter() - t0
    log(f"CMD 종료 코드={proc.returncode}, 소요={dt:.2f}s")
    return proc.returncode, dt


def detect_gpu_backend() -> GpuBackend:
    try:
        import cupy as cp  # type: ignore

        n = cp.cuda.runtime.getDeviceCount()
        if n > 0:
            props = cp.cuda.runtime.getDeviceProperties(0)
            name = props.get("name", b"gpu")
            if isinstance(name, bytes):
                name = name.decode(errors="ignore")
            return GpuBackend(name=f"cupy:{name}", available=True, reason="CUDA device detected")
        return GpuBackend(name="cupy", available=False, reason="no cuda device")
    except Exception as e:
        cupy_reason = str(e)

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return GpuBackend(name=f"torch:{gpu_name}", available=True, reason="torch cuda available")
        return GpuBackend(name="torch", available=False, reason="torch cuda unavailable")
    except Exception as e:
        return GpuBackend(name="none", available=False, reason=f"cupy err={cupy_reason} / torch err={e}")


def parse_tripinfo_streaming(tripinfo_path: Path) -> Tuple[List[float], List[float], List[float]]:
    """대용량 XML 메모리 절약 파싱."""
    durations: List[float] = []
    route_lengths: List[float] = []
    waiting_times: List[float] = []

    log(f"tripinfo 파싱 시작: {tripinfo_path}")
    t0 = time.perf_counter()
    n = 0
    for event, elem in ET.iterparse(tripinfo_path, events=("end",)):
        if elem.tag == "tripinfo":
            n += 1
            durations.append(float(elem.attrib.get("duration", 0.0)))
            route_lengths.append(float(elem.attrib.get("routeLength", 0.0)))
            waiting_times.append(float(elem.attrib.get("waitingTime", 0.0)))
            elem.clear()

    dt = time.perf_counter() - t0
    log(f"tripinfo 파싱 완료: {n}건, {dt:.2f}s")
    return durations, route_lengths, waiting_times


def gpu_compute(durations: List[float], route_lengths: List[float], waiting_times: List[float]) -> Dict[str, float]:
    # 1순위: cupy
    try:
        import cupy as cp  # type: ignore

        t0 = time.perf_counter()
        d = cp.asarray(durations, dtype=cp.float32)
        rl = cp.asarray(route_lengths, dtype=cp.float32)
        wt = cp.asarray(waiting_times, dtype=cp.float32)
        speed = cp.where(d > 0, rl / d, 0)

        out = {
            "n": float(d.size),
            "duration_mean": float(cp.mean(d).get()),
            "duration_p95": float(cp.percentile(d, 95).get()),
            "waiting_mean": float(cp.mean(wt).get()),
            "speed_mean": float(cp.mean(speed).get()),
            "gpu_elapsed_sec": time.perf_counter() - t0,
        }
        return out
    except Exception as e:
        warn(f"cupy 연산 실패, torch 시도: {e}")

    # 2순위: torch
    try:
        import torch  # type: ignore

        device = torch.device("cuda")
        t0 = time.perf_counter()
        d = torch.tensor(durations, dtype=torch.float32, device=device)
        rl = torch.tensor(route_lengths, dtype=torch.float32, device=device)
        wt = torch.tensor(waiting_times, dtype=torch.float32, device=device)
        speed = torch.where(d > 0, rl / d, torch.zeros_like(d))

        q95 = torch.quantile(d, 0.95)
        out = {
            "n": float(d.numel()),
            "duration_mean": float(d.mean().item()),
            "duration_p95": float(q95.item()),
            "waiting_mean": float(wt.mean().item()),
            "speed_mean": float(speed.mean().item()),
            "gpu_elapsed_sec": time.perf_counter() - t0,
        }
        return out
    except Exception as e:
        fail(f"CUDA 연산 실패(cupy/torch 모두 실패): {e}")

    return {}


def cpu_compute(durations: List[float], route_lengths: List[float], waiting_times: List[float]) -> Dict[str, float]:
    import numpy as np

    t0 = time.perf_counter()
    d = np.asarray(durations, dtype=np.float32)
    rl = np.asarray(route_lengths, dtype=np.float32)
    wt = np.asarray(waiting_times, dtype=np.float32)
    speed = np.where(d > 0, rl / d, 0)

    return {
        "n": float(d.size),
        "duration_mean": float(d.mean()),
        "duration_p95": float(np.percentile(d, 95)),
        "waiting_mean": float(wt.mean()),
        "speed_mean": float(speed.mean()),
        "cpu_elapsed_sec": time.perf_counter() - t0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gangnam4 SUMO + CUDA post-processing benchmark")
    parser.add_argument("--sumocfg", default="./map_import/sim_100k_sorted.sumocfg")
    parser.add_argument("--tripinfo", default="./map_import/tripinfo_100k_sorted.xml")
    parser.add_argument("--summary", default="./map_import/summary_100k_sorted.xml")
    parser.add_argument("--sumo-log", default="./map_import/sumo_100k_sorted.log")
    parser.add_argument("--run-sumo", action="store_true", help="SUMO를 먼저 실행")
    parser.add_argument("--sumo-bin", default="sumo")
    parser.add_argument("--require-cuda", action="store_true", help="CUDA 미탐지 시 즉시 실패")
    parser.add_argument("--wait-tripinfo-complete", action="store_true", default=True, help="tripinfo XML 종료 태그까지 대기")
    parser.add_argument("--no-wait-tripinfo-complete", action="store_false", dest="wait_tripinfo_complete", help="tripinfo 완료 대기 비활성화")
    parser.add_argument("--tripinfo-timeout-sec", type=int, default=7200, help="tripinfo 완료 대기 타임아웃(초)")
    parser.add_argument("--run-id", default=None, help="실행 고유 ID(미지정 시 현재 시각으로 자동 생성)")
    parser.add_argument("--output-dir", default="./map_import", help="실행별 출력 파일 저장 디렉토리")
    parser.add_argument("--stat-output", default="./map_import/stat_100k_sorted.xml", help="statistic 출력 파일 경로")
    parser.add_argument("--isolate-outputs", action="store_true", default=True, help="실행별 고유 출력 파일 사용")
    parser.add_argument("--no-isolate-outputs", action="store_false", dest="isolate_outputs", help="고유 출력 파일 비활성화")
    args = parser.parse_args()

    cfg = Path(args.sumocfg)
    tripinfo = Path(args.tripinfo)
    summary = Path(args.summary)
    sumo_log = Path(args.sumo_log)
    stat_output = Path(args.stat_output)

    run_id = args.run_id
    if args.run_sumo and args.isolate_outputs:
        if run_id is None:
            run_id = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tripinfo = output_dir / f"tripinfo_{run_id}.xml"
        summary = output_dir / f"summary_{run_id}.xml"
        stat_output = output_dir / f"stat_{run_id}.xml"
        sumo_log = output_dir / f"sumo_{run_id}.log"
        runtime_cfg = output_dir / f"sim_{run_id}.sumocfg"
        create_runtime_sumocfg(cfg, runtime_cfg, summary, tripinfo, stat_output)
        cfg = runtime_cfg
        log(f"실행ID={run_id} 고유 출력 경로 적용")
        log(f"  tripinfo={tripinfo}")
        log(f"  summary={summary}")
        log(f"  stat={stat_output}")
        log(f"  sumo_log={sumo_log}")

    check_paths(cfg, tripinfo, summary)

    backend = detect_gpu_backend()
    log(f"GPU backend 진단: available={backend.available}, name={backend.name}, reason={backend.reason}")
    if args.require_cuda and not backend.available:
        fail("--require-cuda 옵션 활성화 상태에서 CUDA 백엔드를 찾지 못함")

    if args.run_sumo:
        cmd = [
            args.sumo_bin,
            "-c",
            str(cfg),
            "--no-warnings",
            "true",
            "--log",
            str(sumo_log),
        ]
        rc, _ = run_cmd(cmd)
        if rc != 0:
            fail("SUMO 실행 실패")

    if not tripinfo.exists():
        fail(f"tripinfo 파일 없음: {tripinfo}")

    if args.wait_tripinfo_complete:
        wait_for_tripinfo_completion(tripinfo, timeout_sec=args.tripinfo_timeout_sec)

    durations, route_lengths, waiting_times = parse_tripinfo_streaming(tripinfo)
    if not durations:
        fail("tripinfo 레코드 0건")

    cpu_stats = cpu_compute(durations, route_lengths, waiting_times)
    log(f"CPU 통계: {cpu_stats}")

    if backend.available:
        gpu_stats = gpu_compute(durations, route_lengths, waiting_times)
        log(f"GPU 통계: {gpu_stats}")
        speedup = cpu_stats.get("cpu_elapsed_sec", 0.0) / max(gpu_stats.get("gpu_elapsed_sec", 1e-9), 1e-9)
        log(f"후처리 속도비(CPU/GPU): {speedup:.2f}x")
    else:
        warn("CUDA 사용 불가. CPU 통계만 출력")

    # 진단 검증 로그(가정A/B 확인용)
    log("[진단검증] 가정A 확인: sumocfg 존재 여부 및 파일 크기 출력 완료")
    log("[진단검증] 가정B 확인: GPU backend 탐지 결과 출력 완료")
    log("프로그램 종료")


if __name__ == "__main__":
    main()
