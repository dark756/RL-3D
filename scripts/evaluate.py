"""Evaluate a policy on fixed test sets (M3; reused for the agent at M5): success rate with a 95 % Wilson
interval, failure causes, touchdown speeds and attitude, propellant and time, separately for calm and dust-storm
episodes (D17).

    python -m scripts.evaluate --policy classical                          # 500 calm + 500 storm episodes
    python -m scripts.evaluate --policy classical --episodes 100 --procs 4
    python -m scripts.evaluate --policy classical --out runs/eval/classical --log-failures
    python -m scripts.replay runs/eval/classical/failures/<file>.json --watch

Test sets (`TEST_SETS`): episode k of a set uses seed base + k and fixes only the weather (`storm`), so every
other part of the start is drawn as in training. Training (M4) must use seeds below 1,000,000.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from envs.episode_logger import EpisodeLogger
from envs.mars_lander_env import MarsLanderEnv
from lander.config import load_config, parse_set_args
from lander.episode_log import code_version, to_jsonable
from scripts.run_episodes import POLICIES, make_policy, policy_arg

TEST_SETS = {"calm": (1_000_000, False), "storm": (2_000_000, True)}   # name: (first seed, storm)
Z95 = 1.959964                                                        # two-sided 95 % normal quantile


def wilson(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95 % Wilson score interval for a success rate (sound near 0 % and 100 %, unlike the normal one)."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / d
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def run_episode(env, policy_name: str, seed: int, storm: bool) -> dict[str, Any]:
    """Fly one test episode of `env` (a MarsLanderEnv, possibly wrapped); returns one row of results."""
    obs, info = env.reset(seed=seed, options={"storm": storm})
    policy = make_policy(policy_name, env, seed)
    base = env.unwrapped
    e, start = info["episode"], base.last_result.telemetry
    peak_tilt, done = start.tilt_deg, False
    while not done:
        obs, _, terminated, truncated, info = env.step(policy(obs))
        done = terminated or truncated
        if not base.last_result.report.touched_down:
            peak_tilt = max(peak_tilt, info["truth"]["tilt_deg"])
    o, r, tl = info["outcome"], base.last_result.report, base.last_result.telemetry
    state = getattr(policy, "state", None)
    return {
        "seed": seed, "storm": e["storm"], "wind_mps": e["wind_speed_mps"], "slope_deg": e["slope_deg"],
        "friction": e["friction"], "start_height_m": start.pad_altitude, "start_offset_m": math.hypot(*e["position"][:2]),
        "start_descent_mps": -e["velocity"][2], "start_sideways_mps": math.hypot(*e["velocity"][:2]),
        "result": o["result"], "reason": o["reason"], "success": o["success"], "safe": o["safe"], "reward": o["reward"],
        "distance_m": o["distance_m"], "time_s": o["time_s"], "propellant_used_kg": e["propellant_kg"] - tl.propellant_kg,
        "peak_tilt_deg": peak_tilt, "touchdown_down_mps": r.vz_at_contact, "touchdown_side_mps": r.vh_at_contact,
        "touchdown_tilt_deg": r.tilt_at_contact_deg, "touchdown_rate_dps": r.rate_at_contact_dps,
        "fuel_reserve": None if state is None else state.reserve,
    }


def _run_chunk(args: tuple) -> list[dict[str, Any]]:
    """Worker: one environment, a run of episodes of one set (failed ones re-flown with a logger if asked)."""
    policy_name, set_name, indices, env_file, overrides, log_dir = args
    env = MarsLanderEnv(config=load_config(env_file, overrides))
    base, storm = TEST_SETS[set_name]
    rows = []
    for k in indices:
        row = {"set": set_name, "index": k, **run_episode(env, policy_name, base + k, storm)}
        if log_dir and not row["success"]:
            logged = EpisodeLogger(env, Path(log_dir), trajectory=True, prefix=f"{set_name}{k:04d}")
            again = run_episode(logged, policy_name, base + k, storm)   # deterministic: the same episode
            row["log"] = str(logged.last_path) if again["result"] == row["result"] else "not reproduced"
        rows.append(row)
    env.close()
    return rows


def _stats(values, *qs) -> dict[str, float]:
    v = np.asarray([x for x in values if x is not None and math.isfinite(x)], dtype=float)
    if v.size == 0:
        return {}
    out = {"median": float(np.median(v))}
    out.update({f"p{q}": float(np.percentile(v, q)) for q in qs})
    out["max"] = float(v.max())
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per test set: success and safe-landing rates with Wilson intervals, failure causes and distributions."""
    out = {}
    for name in [s for s in TEST_SETS if any(r["set"] == s for r in rows)]:
        rs = [r for r in rows if r["set"] == name]
        n, ok, safe = len(rs), sum(r["success"] for r in rs), sum(r["safe"] for r in rs)
        down = [r for r in rs if r["touchdown_down_mps"] is not None]
        landed = [r for r in rs if r["safe"]]
        reserve = [r["fuel_reserve"] for r in rs if r["fuel_reserve"] is not None]
        out[name] = {
            "episodes": n, "success": ok, "success_rate": ok / n, "success_ci95": wilson(ok, n),
            "safe": safe, "safe_rate": safe / n, "safe_ci95": wilson(safe, n),
            "results": dict(Counter(r["result"] for r in rs)), "failures": dict(Counter(r["reason"] for r in rs if r["reason"])),
            "distance_m": _stats([r["distance_m"] for r in landed], 95),
            "touchdown_down_mps": _stats([r["touchdown_down_mps"] for r in down], 99),
            "touchdown_side_mps": _stats([r["touchdown_side_mps"] for r in down], 99),
            "touchdown_tilt_deg": _stats([r["touchdown_tilt_deg"] for r in down], 99),
            "touchdown_rate_dps": _stats([r["touchdown_rate_dps"] for r in down], 99),
            "propellant_used_kg": _stats([r["propellant_used_kg"] for r in rs], 95),
            "time_s": _stats([r["time_s"] for r in rs], 95),
            "reward_mean": float(np.mean([r["reward"] for r in rs])),
            "fuel_reserve_used": sum(reserve) if reserve else None,
        }
    return out


def evaluate(policy: str = "classical", episodes: int = 500, sets=tuple(TEST_SETS), procs: int | None = None,
             env_file: str | None = None, overrides: dict | None = None, log_dir: str | Path | None = None,
             progress: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run `episodes` of each test set in `procs` processes; returns (rows, summary)."""
    procs = procs or max(1, min(6, (os.cpu_count() or 2) - 2))
    chunk = max(1, min(10, math.ceil(episodes / procs)))
    tasks = [(policy, s, list(range(i, min(i + chunk, episodes))), env_file, overrides, log_dir)
             for s in sets for i in range(0, episodes, chunk)]
    rows: list[dict[str, Any]] = []
    if procs == 1:
        for t in tasks:
            rows += _run_chunk(t)
    else:
        with ProcessPoolExecutor(procs) as ex:
            futures = [ex.submit(_run_chunk, t) for t in tasks]
            for done, f in enumerate(as_completed(futures), 1):
                rows += f.result()
                if progress and done % max(1, len(tasks) // 10) == 0:
                    print(f"  {len(rows)} of {episodes * len(sets)} episodes", flush=True)
    rows.sort(key=lambda r: (list(TEST_SETS).index(r["set"]), r["index"]))
    return rows, summarize(rows)


def _fmt(d: dict, unit: str, digits: int = 2) -> str:
    return ", ".join(f"{k} {v:.{digits}f}" for k, v in d.items()) + f" {unit}" if d else "-"


def print_report(summary: dict[str, Any]) -> None:
    for name, s in summary.items():
        lo, hi = s["success_ci95"]
        slo, shi = s["safe_ci95"]
        print(f"\n[{name}] {s['episodes']} episodes")
        print(f"  success       {s['success']:4d}  {100 * s['success_rate']:5.1f} %   95 % CI [{100 * lo:.1f}, {100 * hi:.1f}] %")
        print(f"  safe landing  {s['safe']:4d}  {100 * s['safe_rate']:5.1f} %   95 % CI [{100 * slo:.1f}, {100 * shi:.1f}] %"
              "  (on or off the pad)")
        print("  failures      " + (", ".join(f"{k} {v}" for k, v in sorted(s["failures"].items())) or "none"))
        print(f"  distance      {_fmt(s['distance_m'], 'm', 1)}  (safe landings)")
        print(f"  touchdown     down {_fmt(s['touchdown_down_mps'], 'm/s')} | sideways {_fmt(s['touchdown_side_mps'], 'm/s')}")
        print(f"                tilt {_fmt(s['touchdown_tilt_deg'], 'deg', 1)} | pitch/yaw rate {_fmt(s['touchdown_rate_dps'], 'deg/s', 1)}")
        print(f"  propellant    {_fmt(s['propellant_used_kg'], 'kg used', 1)}")
        print(f"  time          {_fmt(s['time_s'], 's', 1)}")
        print(f"  reward        mean {s['reward_mean']:.1f}"
              + ("" if s["fuel_reserve_used"] is None else f"   |   fuel reserve used in {s['fuel_reserve_used']} episodes"))


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="classical", type=policy_arg,
                    help=f"{', '.join(POLICIES)} or ppo:FOLDER, a checkpoint from scripts.train (default classical)")
    ap.add_argument("--episodes", type=int, default=500, help="episodes per test set")
    ap.add_argument("--sets", default=",".join(TEST_SETS), help="comma-separated: " + ", ".join(TEST_SETS))
    ap.add_argument("--procs", type=int, default=None, help="worker processes (default: up to 6, two cores spare)")
    ap.add_argument("--out", default=None, metavar="DIR", help="write summary.json and episodes.csv here")
    ap.add_argument("--log-failures", action="store_true", help="also log every failed episode to DIR/failures")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a fact for this run")
    ap.add_argument("--env-file", default=None, help="fact file to load instead of ./.env")
    args = ap.parse_args(argv)
    sets = tuple(s.strip() for s in args.sets.split(",") if s.strip())
    unknown = set(sets) - set(TEST_SETS)
    if unknown:
        ap.error(f"unknown test sets {sorted(unknown)}")
    if args.log_failures and not args.out:
        ap.error("--log-failures needs --out")
    overrides = parse_set_args(args.set)
    load_config(args.env_file, overrides)                               # fail fast on a bad --set
    out = Path(args.out) if args.out else None
    t0 = time.time()
    print(f"{args.policy}: {args.episodes} episodes per set ({', '.join(sets)})", flush=True)
    rows, summary = evaluate(args.policy, args.episodes, sets, args.procs, args.env_file, overrides,
                             out / "failures" if out and args.log_failures else None, progress=True)
    print(f"done in {time.time() - t0:.0f} s")
    print_report(summary)
    if out:
        out.mkdir(parents=True, exist_ok=True)
        meta = {"policy": args.policy, "episodes_per_set": args.episodes, "test_sets": {s: TEST_SETS[s] for s in sets},
                "overrides": overrides, "code": code_version(), "seconds": time.time() - t0}
        (out / "summary.json").write_text(json.dumps(to_jsonable({**meta, "summary": summary}), indent=1), encoding="utf-8")
        with open(out / "episodes.csv", "w", newline="", encoding="utf-8") as f:
            keys = list(dict.fromkeys(k for r in rows for k in r))
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out / 'summary.json'} and {out / 'episodes.csv'}")
    return summary


if __name__ == "__main__":
    main()
