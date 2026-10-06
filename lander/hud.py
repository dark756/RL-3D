"""Heads-up display text (brief section 8), shared by the live viewer and the video export."""

from __future__ import annotations

import numpy as np

from .visuals import wind_heading_deg


def status_text(res) -> str:
    """One-line flight / landing status from a StepResult."""
    r, tl = res.report, res.telemetry
    if r.done:
        if r.success:
            return "LANDED - all section 7 checks pass"
        if r.failures() == ["engines_off"]:
            return "DOWN - engines still firing (must be cut)"
        if r.leg_broken:
            return f"FAILED: leg broken at {r.vz_at_contact:.2f} m/s down, {r.vh_at_contact:.2f} m/s side"
        return "FAILED: " + ", ".join(r.failures())
    if r.touched_down:
        return f"TOUCHDOWN {r.vz_at_contact:.2f} m/s down, {r.vh_at_contact:.2f} m/s side"
    if tl.engines_off:
        return "ENGINES OFF"
    return "POWERED DESCENT"


def hud_lines(sim, res) -> list[tuple[str, str]]:
    """(label, value) rows: altitude (radar and true), speeds, tilt, spin, propellant, thrust, engines, wind, time."""
    tl, sn = res.telemetry, res.sensors
    tmax = sim.cfg.engine.thrust_max_n
    radar_txt = f"{sn.altitude:8.1f} m" if sn.altitude_valid else "   -- (no lock / <41 m)"
    thr = "  ".join(f"{100 * t / tmax:3.0f}%" for t in tl.thrust_n)
    wind = np.linalg.norm(tl.wind[:2])
    wind_txt = f"{wind:5.1f} m/s toward {wind_heading_deg(tl.wind):5.1f} deg" if wind > 0 else "  calm"
    return [
        ("Time", f"{tl.t:8.2f} s"),
        ("Altitude radar", radar_txt),
        ("Altitude true (pads)", f"{tl.pad_altitude:8.2f} m"),
        ("Vertical speed", f"{tl.vz:+8.2f} m/s"),
        ("Horizontal speed", f"{tl.vh:8.2f} m/s"),
        ("Tilt", f"{tl.tilt_deg:8.2f} deg"),
        ("Spin", f"{tl.spin_dps:+8.2f} deg/s"),
        ("Propellant", f"{tl.propellant_kg:8.2f} kg"),
        ("Thrust 1 2 3 (% max)", thr),
        ("Engines", "CUT" if tl.engines_off else "ON"),
        ("Wind", wind_txt),
        ("Status", status_text(res)),
    ]


def hud_columns(sim, res) -> tuple[str, str]:
    """The HUD as two newline-joined columns (labels, values) for MuJoCo's text overlay."""
    rows = hud_lines(sim, res)
    return "\n".join(r[0] for r in rows), "\n".join(r[1] for r in rows)
