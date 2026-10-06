"""Render a scenario to MP4 without opening a window (headless; Linux/Colab: set MUJOCO_GL=egl).

    python -m scripts.export_video --scenario drop
    python -m scripts.export_video --scenario slope45 --camera ground --out runs/videos/slope.mp4
    python -m scripts.export_video --scenario hover --wind 25,90
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lander.scenarios import SCENARIOS, make_scenario, run_scenario
from lander.sim import LanderSim
from lander.video import VideoWriter
from scripts.common import add_config_args, add_wind_arg, config_from_args, summarize, wind_options


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="drop", choices=sorted(SCENARIOS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output .mp4 (default runs/videos/<scenario>_seed<N>.mp4)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--camera", default="chase", choices=["chase", "ground", "padcam"])
    ap.add_argument("--no-hud", action="store_true")
    add_wind_arg(ap)
    add_config_args(ap)
    args = ap.parse_args(argv)

    sim = LanderSim(config_from_args(args), seed=args.seed)
    sc = make_scenario(args.scenario, sim, seed=args.seed)
    sc.options.update(wind_options(args))
    out = Path(args.out or f"runs/videos/{sc.name}_seed{args.seed}.mp4")
    video = VideoWriter(sim, out, args.width, args.height, args.camera, hud=not args.no_hud)
    try:
        results = run_scenario(sim, sc, on_step=lambda s, res: video.add(res))
    finally:
        video.close()
    print(f"{sc.name}: {sc.description}")
    print(summarize(results[-1].report, results[-1].telemetry))
    print(f"wrote {video.count} frames ({video.count / sim.cfg.sim.control_hz:.1f} s) to {out}")
    return out


if __name__ == "__main__":
    main()
