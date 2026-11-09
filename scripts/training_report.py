"""Summarise a training run (M4): where every seed stands, and the learning curves as a picture.

    python -m scripts.training_report runs/train/<run>          # table + <run>/learning_curves.png

Reads each seed's curriculum.csv (level and success rate after every update) and validation.csv (the policy's
mean action on fixed level-1 episodes, half calm, half storm), both written by scripts/train.py.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def _value(text: str) -> float | str:
    """A CSV cell as a number when it is one (empty = NaN), else the text."""
    if text in ("", "nan"):
        return math.nan
    try:
        return float(text)
    except ValueError:
        return text


def read_rows(path: Path) -> list[dict[str, float | str]]:
    """A CSV written by scripts/train.py as a list of rows, numbers as floats ([] if it does not exist yet)."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [{k: _value(v) for k, v in row.items()} for row in csv.DictReader(f)]


def seed_dirs(run_dir: Path) -> list[Path]:
    """The run's seed folders, in seed order."""
    return sorted((p for p in Path(run_dir).glob("seed*") if p.is_dir()), key=lambda p: int(p.name[4:]))


def seed_status(seed_dir: Path) -> dict[str, float]:
    """The latest training numbers and the latest and best validation of one seed."""
    cur, val = read_rows(seed_dir / "curriculum.csv"), read_rows(seed_dir / "validation.csv")
    last = cur[-1] if cur else {}
    out = {"steps": last.get("timesteps", 0.0), "hours": last.get("hours", 0.0), "level": last.get("level", 0.0),
           "success_at_level": last.get("success_at_level", math.nan), "episodes": last.get("episodes", 0.0),
           "endings": last.get("recent_endings", ""),
           "val_success": math.nan, "val_calm": math.nan, "val_storm": math.nan, "best_val": math.nan}
    if val:
        out.update(val_success=val[-1]["success"], val_calm=val[-1]["success_calm"],
                   val_storm=val[-1]["success_storm"], best_val=max(r["success"] for r in val))
    return out


def print_report(run_dir: str | Path) -> None:
    """One line per seed, then write the learning curves next to the run."""
    run_dir = Path(run_dir)
    print(f"\n{run_dir}")
    print(f"{'seed':>5} {'M steps':>8} {'hours':>6} {'level':>6} {'success here':>13} {'episodes':>9} "
          f"{'validation (calm / storm)':>27} {'best':>6}")
    for d in seed_dirs(run_dir):
        s = seed_status(d)
        val = (f"{s['val_success']:.0%} ({s['val_calm']:.0%} / {s['val_storm']:.0%})"
               if not math.isnan(s["val_success"]) else "-")
        best = f"{s['best_val']:.0%}" if not math.isnan(s["best_val"]) else "-"
        here = f"{s['success_at_level']:.0%}" if not math.isnan(s["success_at_level"]) else "-"
        print(f"{d.name[4:]:>5} {s['steps'] / 1e6:8.2f} {s['hours']:6.2f} {s['level']:6.1f} {here:>13} "
              f"{s['episodes']:9.0f} {val:>27} {best:>6}")
        if s["endings"]:
            print(f"{'':>5} last 100 episodes ended: {s['endings']}")
    png = plot_curves(run_dir)
    if png:
        print(f"learning curves: {png}")


def plot_curves(run_dir: str | Path) -> Path | None:
    """learning_curves.png: curriculum level, success at the level, validation success; one line per seed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    dirs = seed_dirs(run_dir)
    if not any((d / "curriculum.csv").exists() for d in dirs):
        return None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for d in dirs:
        cur, val = read_rows(d / "curriculum.csv"), read_rows(d / "validation.csv")
        steps = [r["timesteps"] / 1e6 for r in cur]
        axes[0].plot(steps, [r["level"] for r in cur], label=d.name)
        axes[1].plot(steps, [100 * r["success_at_level"] for r in cur], label=d.name)
        if val:
            axes[2].plot([r["timesteps"] / 1e6 for r in val], [100 * r["success"] for r in val], marker="o",
                         label=d.name)
    titles = ("Curriculum level (1 = full start range)", "Success at the current level, %",
              "Validation success at level 1, %")
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xlabel("million steps")
        ax.grid(alpha=0.3)
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_ylim(-3, 103)
    axes[2].set_ylim(-3, 103)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    png = run_dir / "learning_curves.png"
    fig.savefig(png, dpi=110)
    plt.close(fig)
    return png


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="a run folder written by scripts.train")
    print_report(ap.parse_args(argv).run)


if __name__ == "__main__":
    main()
