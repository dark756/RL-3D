"""Helpers shared by the command-line scripts: config and wind flags, and a readable landing summary."""

from __future__ import annotations

import argparse

from lander.config import Config, load_config, parse_set_args


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Add --set KEY=VALUE (repeatable) and --env-file to a parser."""
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a fact from .env for this run (recorded as status 'override')")
    parser.add_argument("--env-file", default=None, help="fact file to load instead of ./.env")


def config_from_args(args: argparse.Namespace) -> Config:
    """Load the config with any --set overrides applied."""
    return load_config(args.env_file, parse_set_args(args.set), origin="cli")


def add_wind_arg(parser: argparse.ArgumentParser) -> None:
    """Add --wind SPEED[,DIRECTION]: a steady wind for the episode (calm by default)."""
    parser.add_argument("--wind", default=None, metavar="SPEED[,DIR]",
                        help="steady wind in m/s, blowing toward DIR degrees from +X toward +Y (default 0); "
                             "Mars dust storms are 17-30 m/s (MARS_WIND_STORM_MPS)")


def wind_options(args: argparse.Namespace) -> dict:
    """Reset options for --wind (empty when the flag is not given)."""
    if not args.wind:
        return {}
    parts = [float(p) for p in args.wind.split(",")]
    opts = {"wind_speed_mps": parts[0]}
    if len(parts) > 1:
        opts["wind_direction_deg"] = parts[1]
    return opts


def summarize(report, telemetry) -> str:
    """Multi-line, human-readable outcome of an episode."""
    r = report
    lines = [f"t = {telemetry.t:.2f} s, propellant left {telemetry.propellant_kg:.2f} kg, "
             f"engines {'cut' if telemetry.engines_off else 'ON'}"]
    if not r.touched_down:
        lines.append("no touchdown")
        return "\n".join(lines)
    lines += [
        f"touchdown at {r.first_contact_time:.2f} s: {r.vz_at_contact:.2f} m/s down, "
        f"{r.vh_at_contact:.2f} m/s sideways, tilt {r.tilt_at_contact_deg:.1f} deg"
        + ("  -> LEG BROKEN" if r.leg_broken else ""),
        f"min stability distance {r.min_stability_m:.3f} m, settled: {r.settled}, "
        f"tipped over: {r.tipped_over}, body strike: {r.body_strike}",
        "criteria: " + ", ".join(f"{k}={'n/a (M2)' if v is None else v}" for k, v in r.criteria.items()),
        "RESULT: " + ("SUCCESS" if r.success else ("FAILED: " + ", ".join(r.failures()) if r.done else "undecided")),
    ]
    return "\n".join(lines)
