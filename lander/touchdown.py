"""Touchdown judging (numpy only, D1): the brief's section 7 criteria for the simplified lander (D15).

The legs are rigid. A leg breaks if the first footpad contact is faster than
VIKING_REQ_VERTICAL_SPEED_MPS downward or VIKING_REQ_HORIZONTAL_SPEED_MPS sideways (Viking's leg design
limits, S5). Stability follows TN D-7959 (S4 p9-12): the stability distance (CG to the vertical
planes through each pair of footpads; negative = tipping over) and settling (resting on all three
footpads with kinetic energy below 0.5 % of the impact energy). The agent shuts the engines off itself
(brief D7), and a landing only counts once they are off. Tilt and pad-radius limits are Open decision
O3 (M2): reported, not judged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import Config

SETTLE_MIN_IMPACT_KE_J = 100.0   # assumed floor so very soft touchdowns can still be judged settled
SETTLE_HOLD_S = 0.5              # assumed: KE must stay under the threshold this long (ignores rebound apexes)


def stability_distance(cg: np.ndarray, pads: np.ndarray) -> float:
    """Smallest signed horizontal distance from the CG to the vertical planes through two footpads.

    Positive = CG inside the footpad triangle (stabilising), negative = outside (tipping over).
    Gravity is along -Z (world frame), so the test is done in the X-Y plane.
    """
    c = cg[:2]
    best = math.inf
    for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        a, b, third = pads[i, :2], pads[j, :2], pads[k, :2]
        edge = b - a
        length = math.hypot(edge[0], edge[1])
        if length < 1e-9:
            continue
        cross_c = edge[0] * (c[1] - a[1]) - edge[1] * (c[0] - a[0])
        cross_k = edge[0] * (third[1] - a[1]) - edge[1] * (third[0] - a[0])
        best = min(best, math.copysign(1.0, cross_k) * cross_c / length)
    return best


@dataclass
class TouchdownReport:
    touched_down: bool = False
    first_contact_time: float | None = None
    vz_at_contact: float | None = None          # downward speed (m/s, positive = descending)
    vh_at_contact: float | None = None          # horizontal speed (m/s)
    tilt_at_contact_deg: float | None = None
    leg_broken: bool = False                    # first contact faster than a leg design limit
    min_stability_m: float = math.inf
    settled: bool = False
    settle_time: float | None = None
    tipped_over: bool = False
    body_strike: bool = False
    propellant_exhausted_airborne: bool = False
    engines_off: bool = False                   # engines shut down (must be true when the episode ends)
    final_tilt_deg: float | None = None
    final_xy: tuple[float, float] | None = None
    criteria: dict[str, bool | None] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        """The landing is decided: settled, tipped over, a broken leg, or the body hit the ground."""
        return self.settled or self.tipped_over or self.leg_broken or (self.body_strike and self.touched_down)

    @property
    def success(self) -> bool:
        return self.done and all(v for v in self.criteria.values() if v is not None)

    def failures(self) -> list[str]:
        return [k for k, v in self.criteria.items() if v is False]


class TouchdownMonitor:
    """Watches the physics steps and fills a `TouchdownReport`."""

    def __init__(self, cfg: Config):
        self.req = cfg.req
        self.report = TouchdownReport()
        self._impact_ke = 0.0
        self._calm_since: float | None = None

    def reset(self) -> None:
        self.report = TouchdownReport()
        self._impact_ke = 0.0
        self._calm_since = None

    def note_engines(self, engines_off: bool) -> None:
        """Record whether the engines are shut down (checked every physics step, even after landing)."""
        if engines_off != self.report.engines_off:
            self.report.engines_off = engines_off
            if self.report.touched_down:
                self._judge()

    def update(self, t: float, pad_contact, body_contact: bool, v_cg: np.ndarray, ke: float,
               stability: float, tilt_deg: float, empty: bool, xy) -> None:
        r = self.report
        if not r.touched_down:
            if empty:
                r.propellant_exhausted_airborne = True
            if body_contact:
                r.body_strike = True
            if not any(pad_contact) and not body_contact:
                return
            r.touched_down = True
            r.first_contact_time = t
            r.vz_at_contact = float(-v_cg[2])
            r.vh_at_contact = float(math.hypot(v_cg[0], v_cg[1]))
            r.tilt_at_contact_deg = float(tilt_deg)
            r.leg_broken = (r.vz_at_contact > self.req.vertical_speed_mps
                            or r.vh_at_contact > self.req.horizontal_speed_mps)
            self._impact_ke = max(ke, SETTLE_MIN_IMPACT_KE_J)
        if r.settled or r.tipped_over or r.leg_broken:
            self._judge()
            if r.leg_broken and r.final_xy is None:
                r.final_tilt_deg, r.final_xy = float(tilt_deg), (float(xy[0]), float(xy[1]))
            return
        r.min_stability_m = min(r.min_stability_m, float(stability))
        r.body_strike = r.body_strike or bool(body_contact)
        if stability < 0.0:
            r.tipped_over = True
        elif all(pad_contact) and ke < self.req.settle_ke_frac * self._impact_ke:   # resting on all three feet
            if self._calm_since is None:
                self._calm_since = t
            elif t - self._calm_since >= SETTLE_HOLD_S:
                r.settled, r.settle_time = True, t
        else:
            self._calm_since = None
        if r.done:
            r.final_tilt_deg = float(tilt_deg)
            r.final_xy = (float(xy[0]), float(xy[1]))
        self._judge()

    def _judge(self) -> None:
        r, q = self.report, self.req
        r.criteria = {
            "vertical_speed": r.vz_at_contact is not None and r.vz_at_contact <= q.vertical_speed_mps,
            "horizontal_speed": r.vh_at_contact is not None and r.vh_at_contact <= q.horizontal_speed_mps,
            "stable": (not r.tipped_over) if (r.settled or r.tipped_over) else None,
            "no_body_strike": not r.body_strike,
            "engines_off": r.engines_off,
            "propellant_left_at_touchdown": not r.propellant_exhausted_airborne,
            "inside_pad_radius": None,     # O3, decided at M2
            "tilt_within_limit": None,     # O3, decided at M2 (S3 Table 8 suggests 5 deg)
        }
