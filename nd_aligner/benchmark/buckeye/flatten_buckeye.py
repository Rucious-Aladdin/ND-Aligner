"""
Flatten the Buckeye Corpus into a single directory.

The distributed corpus nests recordings under speaker directories and leaves
`.zip.extracted` markers behind; the annotation patches are written against the
flat layout.

    python flatten_buckeye.py /path/to/buckeye /path/to/flat
    cd /path/to/flat
    git init && git add . && git commit -m "Initial commit"
    git apply /path/to/buckeye_fixes.patch
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

RECORDING = re.compile(r"^s\d{4}[ab]$")
EXTENSIONS = {".wav", ".words", ".phones", ".txt", ".log"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])  # pyright: ignore
    parser.add_argument("source", type=Path, help="corpus root holding s01 ... s40")
    parser.add_argument("destination", type=Path, help="directory to write into")
    args = parser.parse_args()

    args.destination.mkdir(parents=True, exist_ok=True)

    count = 0
    for path in sorted(args.source.rglob("*")):
        if not path.is_file() or path.suffix not in EXTENSIONS:
            continue
        if not RECORDING.match(path.stem):
            continue

        target = args.destination / path.name
        if target.exists():
            raise SystemExit(f"Already exists, refusing to overwrite: {target}")

        shutil.copy2(path, target)
        count += 1

    print(f"Copied {count} files to {args.destination}")


if __name__ == "__main__":
    main()
