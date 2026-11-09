"""The evaluation script (M3, reused at M5): Wilson intervals, the fixed test sets, summaries, failure logs."""

from __future__ import annotations

import pytest

from scripts.evaluate import TEST_SETS, evaluate, wilson
from scripts.replay import replay


def test_wilson_interval():
    lo, hi = wilson(95, 100)
    assert lo == pytest.approx(0.8882, abs=5e-4) and hi == pytest.approx(0.9785, abs=5e-4)
    lo, hi = wilson(500, 500)                       # no failures in 500: still not certain
    assert hi == 1.0 and lo == pytest.approx(0.9924, abs=5e-4)
    assert wilson(0, 10)[0] == 0.0 and wilson(0, 0) == (0.0, 1.0)


def test_fixed_test_sets_summary_and_failure_logs(tmp_path):
    rows, summary = evaluate("random", episodes=2, procs=1, log_dir=tmp_path)    # random actions fail fast
    (calm, _), (storm, _) = TEST_SETS["calm"], TEST_SETS["storm"]
    assert [r["seed"] for r in rows] == [calm, calm + 1, storm, storm + 1]
    assert [r["storm"] for r in rows] == [False, False, True, True]
    s = summary["calm"]
    assert s["episodes"] == 2 and s["success"] == 0 and s["success_ci95"] == wilson(0, 2)
    assert sum(s["failures"].values()) == 2 and s["fuel_reserve_used"] is None
    assert replay(rows[0]["log"]).identical                                     # a logged failure replays
