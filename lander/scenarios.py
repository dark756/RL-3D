"""Scripted open-loop scenarios (no controller, no RL) for tests, the viewer, video export and benchmarks.

The only 'policy' used is hover trim: valve settings that make total thrust equal to weight with zero
pitch/yaw torque about the CG, plus cutting the engines the moment a footpad touches (the agent's
job from M2 on). Nothing corrects attitude or drift, so these runs show the raw physics: free fall,
drift from the thrust misalignment, touchdown, broken legs, tip-over.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .sim import InitialState, LanderSim, StepResult
from .vehicle import Command

LOW_PROPELLANT_KG = 3.0        # assumed: enough for the few seconds these near-ground scenarios last


def hover(sim: LanderSim) -> Command:
    """Thrust = weight, engines never cut."""
    return Command(sim.hover_trim())


def hover_then_cut(sim: LanderSim) -> Command:
    """Thrust = weight until the first footpad contact, then engines off for good.

    Uses the latched touchdown flag: rigid legs rebound slightly (~13 % of the impact speed), so the
    feet may already be off the ground again at the next control step.
    """
    return Command(sim.hover_trim(), engines_off=sim.monitor.report.touched_down)


@dataclass
class Scenario:
    name: str
    description: str
    duration_s: float
    initial_state: InitialState
    options: dict = field(default_factory=dict)
    policy: Callable[[LanderSim], Command] = hover_then_cut
    seed: int = 0


SCENARIOS = {
    "drop": "Viking-like touchdown: 6 m up, 2.44 m/s down, thrust = weight, engines cut on contact -> lands",
    "hard_drop": "4 m/s down + 0.6 m/s sideways: faster than the 3.34 m/s leg limit -> a leg breaks",
    "slope45": "2.44 m/s touchdown on a 45 deg slope -> usually tips over (S4: p = 0.85)",
    "hover": "30 m up, thrust = weight, no attitude control -> the thrust misalignment tilts it and it drifts",
    "freefall": "engines off from 100 m -> hits at ~27 m/s and breaks its legs",
}


def make_scenario(name: str, sim: LanderSim, seed: int = 0) -> Scenario:
    """Build a named scenario for this simulator (initial state depends on the configured vehicle)."""
    if name == "drop":
        st = sim.state_above_ground(6.0, velocity=(0.0, 0.0, -2.44), propellant_kg=LOW_PROPELLANT_KG + 3.0)
        return Scenario(name, SCENARIOS[name], 6.0, st, {}, seed=seed)
    if name == "hard_drop":
        st = sim.state_above_ground(1.0, velocity=(0.6, 0.0, -4.0), propellant_kg=LOW_PROPELLANT_KG)
        return Scenario(name, SCENARIOS[name], 4.0, st, {}, seed=seed)
    if name == "slope45":
        opts = {"slope_deg": 45.0, "slope_direction_deg": 0.0}
        st = sim.state_above_ground(1.0, velocity=(0.0, 0.0, -2.44), propellant_kg=LOW_PROPELLANT_KG,
                                    slope_deg=45.0, slope_direction_deg=0.0)
        return Scenario(name, SCENARIOS[name], 6.0, st, opts, seed=seed)
    if name == "hover":
        st = sim.state_above_ground(30.0)
        return Scenario(name, SCENARIOS[name], 20.0, st, {}, policy=hover, seed=seed)
    if name == "freefall":
        st = sim.state_above_ground(100.0, velocity=(0.0, 0.0, 0.0))
        return Scenario(name, SCENARIOS[name], 10.0, st, {"engines_enabled": False}, seed=seed)
    raise KeyError(f"unknown scenario '{name}'; choose from {sorted(SCENARIOS)}")


def run_scenario(sim: LanderSim, scenario: Scenario,
                 on_step: Callable[[LanderSim, StepResult], None] | None = None,
                 settle_extra_s: float = 1.0) -> list[StepResult]:
    """Reset and run a scenario until its duration, or `settle_extra_s` after the landing is decided."""
    results = [sim.reset(seed=scenario.seed, initial_state=scenario.initial_state, options=scenario.options)]
    if on_step is not None:
        on_step(sim, results[0])
    done_at = None
    while sim.t < scenario.duration_s - 1e-9:
        res = sim.step(scenario.policy(sim))
        results.append(res)
        if on_step is not None:
            on_step(sim, res)
        if res.report.done and done_at is None:
            done_at = sim.t
        if done_at is not None and sim.t >= done_at + settle_extra_s:
            break
    return results
