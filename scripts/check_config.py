"""Print every fact from .env with unit, status and source, plus the derived sanity numbers.

    python -m scripts.check_config                  # everything
    python -m scripts.check_config --status assumed # only the values that still need checking
    python -m scripts.check_config --grep RADAR     # keys containing RADAR
    python -m scripts.check_config --dump-mjcf runs/viking.xml
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from lander.config import GROUPS, derived
from lander.geometry import build_geometry
from lander.mjcf import build_mjcf
from scripts.common import add_config_args, config_from_args


def format_value(value) -> str:
    if isinstance(value, tuple):
        return ",".join(f"{v:g}" for v in value)
    return f"{value:g}" if isinstance(value, float) else str(value)


def print_table(cfg, status: str | None = None, grep: str | None = None) -> None:
    """Facts grouped by section; optionally filtered by status or key substring."""
    for prefix, name, _ in GROUPS:
        rows = [p for k, p in cfg.params.items()
                if k.startswith(prefix) and not any(k.startswith(o) for o, _, _ in GROUPS
                                                    if o != prefix and o.startswith(prefix))]
        rows = [p for p in rows if (status is None or p.status == status) and (grep is None or grep in p.key)]
        if not rows:
            continue
        print(f"\n[{name}]  {prefix}*")
        for p in rows:
            note = f"  -- {p.note}" if p.note else ""
            print(f"  {p.key:<34} {format_value(p.value):>22} {p.unit:<7} {p.status:<9} {p.source}{note}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", choices=["published", "derived", "assumed", "override"])
    ap.add_argument("--grep", help="only keys containing this text")
    ap.add_argument("--dump-mjcf", metavar="PATH", help="also write the generated MuJoCo XML here")
    add_config_args(ap)
    args = ap.parse_args(argv)
    cfg = config_from_args(args)

    print(f"Facts from {cfg.env_path}")
    print_table(cfg, args.status, args.grep)
    if args.status is None and args.grep is None:
        print("\n[derived]  computed from the facts above (compare with the brief, section 4)")
        for key, (value, unit, formula) in derived(cfg).items():
            print(f"  {key:<34} {value:>12.4g} {unit:<6} = {formula}")
    counts = Counter(p.status for p in cfg.params.values())
    print("\nstatus counts: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    if args.dump_mjcf:
        path = Path(args.dump_mjcf)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(build_mjcf(cfg, build_geometry(cfg)), encoding="utf-8")
        print(f"MJCF written to {path}")


if __name__ == "__main__":
    main()
