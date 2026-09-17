#!/usr/bin/env python3
"""Regenerate checksums.sha256 over the released model and research-data artifacts.

The file lists every artifact a third party should be able to verify. Paths are
collected from the patterns below rather than maintained by hand, so an added or
removed artifact cannot silently drift out of the manifest.

    python -m scripts.write_checksums            # rewrite
    python -m scripts.write_checksums --check    # verify only, non-zero on drift
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

HEADER = [
    "# SHA-256 checksums of the released model and research-data artifacts.",
    "# Verify from the repository root BEFORE rerunning the pipeline:",
    "#   shasum -a 256 -c checksums.sha256",
]

PATTERNS = (
    "models/**/*.json",
    "models/**/*.pkl",
    "models/**/*.pt",
    "results/verification/**/*.csv",
    "results/verification/**/*.json",
    "results/execution_audit/**/*.csv",
    "results/execution_audit/**/*.json",
)

# Regenerable inputs that live under the scanned trees but are not released
# artifacts; matching them would make the manifest machine-specific.
EXCLUDE_PARTS = ("hpo", "__pycache__")


def collect() -> list[Path]:
    seen: set[Path] = set()
    for pattern in PATTERNS:
        seen.update(
            p for p in REPO.glob(pattern)
            if p.is_file() and not any(part in EXCLUDE_PARTS for part in p.parts)
        )
    return sorted(seen, key=lambda p: str(p.relative_to(REPO)))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def render(files: list[Path]) -> str:
    lines = list(HEADER)
    lines += [f"{digest(p)}  {p.relative_to(REPO)}" for p in files]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="compare against the existing file instead of rewriting")
    args = ap.parse_args()

    target = REPO / "checksums.sha256"
    files = collect()
    if not files:
        raise SystemExit("Fail fast: no artifacts matched — wrong working directory?")
    rendered = render(files)

    if args.check:
        current = target.read_text(encoding="utf-8") if target.exists() else ""
        if current == rendered:
            print(f"checksums.sha256 up to date ({len(files)} artifacts)")
            return
        print("checksums.sha256 DIFFERS from the artifacts on disk", file=sys.stderr)
        raise SystemExit(1)

    target.write_text(rendered, encoding="utf-8")
    print(f"checksums.sha256 written ({len(files)} artifacts)")


if __name__ == "__main__":
    main()
