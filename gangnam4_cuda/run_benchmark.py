#!/usr/bin/env python3
"""
통합 성능 벤치마크 — 동일 입력으로 4개 시뮬레이터 일관 측정.

systems:
  sumo       SUMO microscopic (Docker, 단일 스레드)
  macro_cpu  lane-CTM CPU (lane_cpu_simulator_mt.py --model ctm)
  meso_cpu   mesoscopic CPU (meso_sim.py)
  meso_gpu   mesoscopic GPU (meso_gpu.py)  ← 추후 구축 시 자동 인식

측정(시스템당 warmup 1 + repeats N):
  sim_wall   시뮬레이션 루프 wall(로드 제외) — 알고리즘 비교의 공정 지표
  total_wall 엔드투엔드 wall(로드/파싱/전송/기록 포함) — 실사용 turnaround
  rtf        real-time factor = sim_time / sim_wall (높을수록 빠름)
  peak_rss   최대 상주 메모리(MB, /usr/bin/time -v 있을 때)
모두 동일 sim_time(기본 3600s). dt는 엔진별 기본(meso 1.0, CTM 0.5) — step 수도 기록.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_WALL_RE = re.compile(r"완료:\s*([0-9.]+)s")
_SUMO_DUR_RE = re.compile(r"Duration:\s*([0-9.]+)s")
_SUMO_RTF_RE = re.compile(r"Real time factor:\s*([0-9.]+)")
_RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*([0-9]+)")

_HAVE_TIME = shutil.which("/usr/bin/time") is not None


def log(m): print(f"[LOG] {m}")


def _wrap_time(cmd: list[str]) -> list[str]:
    if _HAVE_TIME:
        return ["/usr/bin/time", "-v"] + cmd
    return cmd


def _parse(out: str, patterns: dict) -> dict:
    res = {}
    for k, rx in patterns.items():
        m = rx.search(out)
        res[k] = float(m.group(1)) if m else float("nan")
    return res


def make_sumo_cfg(net: str, route: str, sim_time: float, cfg_path: Path):
    cfg_path.write_text("\n".join([
        "<configuration>",
        f'  <input><net-file value="{net}"/><route-files value="{route}"/></input>',
        f'  <time><begin value="0"/><end value="{int(sim_time)}"/></time>',
        "</configuration>",
    ]), encoding="utf-8")


def build_cmd(system: str, args, tmp: Path) -> tuple[list[str], dict]:
    """(명령, sim 메타) 반환."""
    steps = int(round(args.sim_time / args.dt))
    if system == "sumo":
        ws = Path(args.workspace).resolve()
        # SUMO cfg는 컨테이너에 마운트되도록 workspace 안에 둔다
        cfg = ws / "map_import" / "_bench_sumo.sumocfg"
        net_c = "/workspace/" + str(Path(args.net_file).resolve().relative_to(ws))
        route_c = "/workspace/" + str(Path(args.route_file).resolve().relative_to(ws))
        cfg_c = "/workspace/" + str(cfg.resolve().relative_to(ws))
        make_sumo_cfg(net_c, route_c, args.sim_time, cfg)
        # 컨테이너에 /usr/bin/time 없음 → RSS는 생략(SUMO는 sim_wall/RTF만). SUMO가
        # 자체 "Duration/Real time factor"를 출력하므로 sim-loop wall은 정확히 파싱됨.
        inner = ("sumo -c " + cfg_c +
                 " --no-step-log true --no-warnings true --duration-log.statistics true")
        cmd = ["docker", "run", "--rm", "-v", f"{ws}:/workspace", "-w", "/workspace",
               args.docker_image, "bash", "-lc", inner]
        return cmd, {"steps": "n/a(continuous)", "dt": "n/a"}
    if system == "macro_cpu":
        cmd = [sys.executable, str(_HERE / "lane_cpu_simulator_mt.py"),
               "--net-file", args.net_file, "--route-file", args.route_file,
               "--model", "ctm", "--sim-time", str(args.sim_time), "--dt", str(args.dt),
               "--log-interval", "999999", "--output-csv", str(tmp / "m.csv"),
               "--edge-output-csv", str(tmp / "m.edge.csv")]
        return _wrap_time(cmd), {"steps": steps, "dt": args.dt}
    if system == "meso_cpu":
        cmd = [sys.executable, str(_HERE / "meso_sim.py"),
               "--net-file", args.net_file, "--route-file", args.route_file,
               "--sim-time", str(args.sim_time), "--dt", str(args.meso_dt),
               "--log-interval", "999999", "--trip-output-csv", "",
               "--edge-output-csv", str(tmp / "meso.edge.csv")]
        return _wrap_time(cmd), {"steps": int(round(args.sim_time / args.meso_dt)), "dt": args.meso_dt}
    if system == "meso_gpu":
        gpu = _HERE / "meso_gpu.py"
        if not gpu.exists():
            return None, {}
        ws = Path(args.workspace).resolve()
        net_c = "/workspace/" + str(Path(args.net_file).resolve().relative_to(ws))
        route_c = "/workspace/" + str(Path(args.route_file).resolve().relative_to(ws))
        # 엔진 코드는 _HERE(워크트리)에 있고 /workspace(메인 체크아웃)엔 없을 수 있으므로
        # _HERE를 /workspace/gangnam4_cuda 위에 오버레이 마운트한다.
        edge_out = "/workspace/map_import/_bench_mesogpu.edge.csv"
        inner = (f"python3 /workspace/gangnam4_cuda/meso_gpu.py "
                 f"--net-file {net_c} --route-file {route_c} --sim-time {args.sim_time} "
                 f"--dt {args.meso_dt} --log-interval 999999 --trip-output-csv '' "
                 f"--edge-output-csv {edge_out}")
        cmd = ["docker", "run", "--rm", "--gpus", "all",
               "-v", f"{ws}:/workspace", "-v", f"{_HERE}:/workspace/gangnam4_cuda",
               "-w", "/workspace", args.docker_image, "bash", "-lc", inner]
        return cmd, {"steps": int(round(args.sim_time / args.meso_dt)), "dt": args.meso_dt}
    if system == "meso_gpu_graph":
        gpu = _HERE / "meso_gpu_graph.py"
        if not gpu.exists():
            return None, {}
        ws = Path(args.workspace).resolve()
        net_c = "/workspace/" + str(Path(args.net_file).resolve().relative_to(ws))
        route_c = "/workspace/" + str(Path(args.route_file).resolve().relative_to(ws))
        edge_out = "/workspace/map_import/_bench_mesogpugraph.edge.csv"
        inner = (f"python3 /workspace/gangnam4_cuda/meso_gpu_graph.py "
                 f"--net-file {net_c} --route-file {route_c} --sim-time {args.sim_time} "
                 f"--dt {args.meso_dt} --edge-output-csv {edge_out}")
        cmd = ["docker", "run", "--rm", "--gpus", "all",
               "-v", f"{ws}:/workspace", "-v", f"{_HERE}:/workspace/gangnam4_cuda",
               "-w", "/workspace", args.docker_image, "bash", "-lc", inner]
        return cmd, {"steps": int(round(args.sim_time / args.meso_dt)), "dt": args.meso_dt}
    return None, {}


def run_system(system: str, args, tmp: Path) -> dict:
    cmd, meta = build_cmd(system, args, tmp)
    if cmd is None:
        log(f"[{system}] 미구현 — 스킵")
        return None
    # warmup
    log(f"[{system}] warmup...")
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sim_walls, tot_walls, rss_list, sumo_rtf = [], [], [], []
    for i in range(args.repeats):
        t0 = time.perf_counter()
        pr = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        tot = time.perf_counter() - t0
        out = pr.stdout
        if system == "sumo":
            d = _parse(out, {"dur": _SUMO_DUR_RE, "rtf": _SUMO_RTF_RE})
            sim_walls.append(d["dur"]); sumo_rtf.append(d["rtf"])
        else:
            m = _WALL_RE.search(out)
            sim_walls.append(float(m.group(1)) if m else float("nan"))
        tot_walls.append(tot)
        rm = _RSS_RE.search(out)
        if rm:
            rss_list.append(int(rm.group(1)) / 1024.0)  # MB
        log(f"  rep {i+1}/{args.repeats}: sim={sim_walls[-1]:.3f}s total={tot:.2f}s")

    sim_med = statistics.median([x for x in sim_walls if x == x]) if sim_walls else float("nan")
    tot_med = statistics.median(tot_walls)
    rtf = (args.sim_time / sim_med) if sim_med and sim_med == sim_med and sim_med > 0 else float("nan")
    rss = statistics.median(rss_list) if rss_list else float("nan")
    return {
        "system": system, "steps": meta.get("steps"), "dt": meta.get("dt"),
        "sim_wall_s": sim_med, "total_wall_s": tot_med, "rtf": rtf,
        "peak_rss_mb": rss,
        "sumo_rtf_self": (statistics.median(sumo_rtf) if sumo_rtf else float("nan")),
    }


def main():
    p = argparse.ArgumentParser(description="Unified performance benchmark (SUMO/macro/meso)")
    p.add_argument("--net-file", default="./map_import/gangnam4_generated.net.xml")
    p.add_argument("--route-file", default="./map_import/gangnam4_generated.sorted.rou.xml")
    p.add_argument("--sim-time", type=float, default=3600.0)
    p.add_argument("--dt", type=float, default=0.5, help="macro/CTM dt")
    p.add_argument("--meso-dt", type=float, default=1.0, help="meso dt")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--systems", default="sumo,macro_cpu,meso_cpu")
    p.add_argument("--workspace", default="/home/mgkyung/ts")
    p.add_argument("--docker-image", default="gangnam4-cuda-sumo:latest")
    p.add_argument("--tmp-dir", default="/tmp/bench")
    p.add_argument("--out-csv", default="/tmp/bench/benchmark.csv")
    args = p.parse_args()

    tmp = Path(args.tmp_dir); tmp.mkdir(parents=True, exist_ok=True)
    log(f"벤치마크: net={Path(args.net_file).name} route={Path(args.route_file).name} "
        f"sim_time={args.sim_time}s repeats={args.repeats}  (/usr/bin/time={_HAVE_TIME})")
    rows = []
    for s in args.systems.split(","):
        r = run_system(s.strip(), args, tmp)
        if r:
            rows.append(r)

    out = Path(args.out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["system", "steps", "dt", "sim_wall_s", "total_wall_s", "rtf", "peak_rss_mb", "sumo_rtf_self"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]) for k in cols})

    log("=" * 78)
    log(f"{'system':12s} {'sim_wall':>10s} {'total_wall':>11s} {'RTF':>9s} {'peak_RSS_MB':>12s}")
    base = next((r["sim_wall_s"] for r in rows if r["system"] == "sumo"), None)
    for r in rows:
        sp = f"  ({base/r['sim_wall_s']:.0f}x vs SUMO)" if base and r['sim_wall_s']>0 and r['system']!='sumo' else ""
        rss = f"{r['peak_rss_mb']:.0f}" if r['peak_rss_mb']==r['peak_rss_mb'] else "n/a"
        log(f"{r['system']:12s} {r['sim_wall_s']:>9.3f}s {r['total_wall_s']:>10.2f}s "
            f"{r['rtf']:>8.1f}x {rss:>12s}{sp}")
    log("=" * 78)
    log(f"결과 저장: {out}")


if __name__ == "__main__":
    main()
