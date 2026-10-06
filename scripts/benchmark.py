"""Measure simulator speed on this machine, single process and N processes in parallel.

Used to plan M4 (PPO on CPU, D9): RL throughput is limited by how fast the simulator steps.

    python -m scripts.benchmark                        # 1, 2, 4, 6, 8 processes
    python -m scripts.benchmark --procs 1,4 --sim-seconds 60
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time


def _worker(sim_seconds: float, seed: int) -> tuple[float, float]:
    """Fly the hover scenario (in flight, no contacts) for `sim_seconds`; returns (control steps, wall s)."""
    from lander.scenarios import make_scenario
    from lander.sim import LanderSim

    sim = LanderSim(seed=seed)
    sc = make_scenario("hover", sim, seed=seed)
    steps, flown, t0 = 0, 0.0, time.perf_counter()
    while flown < sim_seconds:
        sim.reset(seed=seed + steps, initial_state=sc.initial_state, options=sc.options)
        while sim.t < sc.duration_s and flown + sim.t < sim_seconds:
            sim.step(sc.policy(sim))
            steps += 1
        flown += sim.t
    return steps, time.perf_counter() - t0


def run(procs: int, sim_seconds: float) -> tuple[float, float]:
    """Aggregate control steps per second and real-time factor with `procs` worker processes."""
    ctx = mp.get_context("spawn")
    with ctx.Pool(procs) as pool:
        out = pool.starmap(_worker, [(sim_seconds, i) for i in range(procs)])
    steps = sum(s for s, _ in out)
    wall = max(w for _, w in out)
    return steps / wall, sim_seconds * procs / wall


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--procs", default="1,2,4,6,8", help="comma-separated process counts")
    ap.add_argument("--sim-seconds", type=float, default=30.0, help="simulated seconds per process")
    args = ap.parse_args(argv)
    print(f"CPU logical cores: {os.cpu_count()}")
    base = None
    for n in [int(x) for x in args.procs.split(",")]:
        rate, rt = run(n, args.sim_seconds)
        base = base or rate
        hours_10m = 10e6 / rate / 3600
        print(f"{n:2d} process(es): {rate:8.0f} control steps/s  ({rt:6.1f}x real time total, "
              f"speed-up {rate / base:4.2f}x)  -> 10M env steps in ~{hours_10m:.1f} h (simulation only)")


if __name__ == "__main__":
    main()
