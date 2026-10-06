"""Download the NASA source PDFs (S3, S4, S5) into docs/sources/ so every .env citation can be checked.

    python -m scripts.fetch_sources
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

SOURCES = {
    "S3_RP-1027_Viking75_design_test_summary_vol1.pdf":
        "https://ntrs.nasa.gov/api/citations/19810001592/downloads/19810001592.pdf",
    "S4_TN-D-7959_Viking_touchdown_MonteCarlo.pdf":
        "https://ntrs.nasa.gov/api/citations/19750023452/downloads/19750023452.pdf",
    "S5_JPL-TM-33-777_Viking_landing_gear.pdf":
        "https://ntrs.nasa.gov/api/citations/19760021205/downloads/19760021205.pdf",
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default="docs/sources")
    ap.add_argument("--force", action="store_true", help="download again even if the file exists")
    args = ap.parse_args(argv)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        path = dest / name
        if path.exists() and not args.force:
            print(f"have {path} ({path.stat().st_size / 1e6:.1f} MB)")
            continue
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path)
        print(f"  -> {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
