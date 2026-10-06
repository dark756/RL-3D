"""Watch a scenario live in the MuJoCo viewer (opens a window). No manual flight: the scenario's
scripted policy flies it (the RL agent will plug in here at M4/M5).

    python -m scripts.watch --scenario drop
    python -m scripts.watch --scenario slope45 --speed 0.25
    python -m scripts.watch --scenario hover --wind 25,90

Keys in the window: Space pause/resume, Enter restart; mouse orbits/zooms; [ ] cycle cameras.
"""

from __future__ import annotations

import argparse

from lander.scenarios import SCENARIOS, make_scenario
from lander.sim import LanderSim
from lander.viewer import watch
from scripts.common import add_config_args, add_wind_arg, config_from_args, summarize, wind_options


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="drop", choices=sorted(SCENARIOS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed, e.g. 0.25 for slow motion")
    ap.add_argument("--max-seconds", type=float, default=None, help="close the window after this many seconds")
    ap.add_argument("--no-hud", action="store_true")
    add_wind_arg(ap)
    add_config_args(ap)
    args = ap.parse_args(argv)

    sim = LanderSim(config_from_args(args), seed=args.seed)
    sc = make_scenario(args.scenario, sim, seed=args.seed)
    sc.options.update(wind_options(args))
    print(f"{sc.name}: {sc.description}")
    last = {}

    def reset_fn():
        last["res"] = sim.reset(seed=sc.seed, initial_state=sc.initial_state, options=sc.options)
        return last["res"]

    def policy(s):
        return sc.policy(s)

    watch(sim, policy, reset_fn, speed=args.speed, max_seconds=args.max_seconds, hud=not args.no_hud)
    print(summarize(sim.monitor.report, sim.telemetry()))


if __name__ == "__main__":
    main()
