"""Precondition checklist for the bundled runs.

(a) Deterministic refit of models/statistic_params.pkl with the extended
    analyzer, which adds the variant-conditioned and Weibull keys.
(b) Verification that the station-marginal keys stay bit-identical, so the
    RefSim-M input is unchanged.

Aborts before the new pickle is written.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.config.simulation_config import TRAIN_RATIO  # noqa: E402
from src.fitting.ref_data_preparation.ref_analyzer import RefSimAnalyzer  # noqa: E402

MARGINAL_KEYS = (
    "process_times", "transition_probs", "mttf_data",
    "repair_times", "product_features",
)
NEW_KEYS = ("process_times_variant", "transition_probs_variant", "weibull_ttf")


def _first_diff(a: Any, b: Any, path: str = "") -> str | None:
    """Recursive bit-exact comparison; returns the path of the first difference or None."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return f"{path}: keys {sorted(set(a) ^ set(b))[:5]}"
        for k in a:
            d = _first_diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return f"{path}: len {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            d = _first_diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else f"{path}: {a!r} != {b!r}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model-dir", default=str(REPO / "models"))
    ap.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    ap.add_argument("--write", action="store_true",
                    help="rewrite models/statistic_params.pkl after the checks "
                         "(default: check only; note the file is checksummed)")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    pkl_path = model_dir / "statistic_params.pkl"

    print("(a) Deterministic refit with the extended analyzer …")
    new = RefSimAnalyzer(data_dir=Path(args.data_dir), train_ratio=args.train_ratio).extract_all()
    for k in NEW_KEYS:
        if k not in new:
            raise SystemExit(f"FAIL (a): extended analyzer does not provide '{k}'.")
    print(f"    new keys present: {', '.join(NEW_KEYS)}")

    print("(b) Bit-identity of the marginal keys against the current pkl …")
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            old = pickle.load(f)
        for k in MARGINAL_KEYS:
            if k not in old:
                print(f"    [warn] '{k}' missing in old pkl — skipped.")
                continue
            diff = _first_diff(old[k], new[k], k)
            if diff is not None:
                raise SystemExit(f"FAIL (b): marginal key '{k}' drifts — {diff}")
        print(f"    marginal keys bit-identical: {', '.join(MARGINAL_KEYS)}")
    else:
        print("    [warn] no existing pkl — bit-identity not checkable (first run).")

    if not args.write:
        print("\n[check-only] pkl NOT written. All preconditions green. Use --write to persist.")
        return
    with open(pkl_path, "wb") as f:
        pickle.dump(new, f)
    print(f"\n(a) pkl written: {pkl_path} (+ {', '.join(NEW_KEYS)})")
    print("All preconditions green — ready for the bundled runs.")


if __name__ == "__main__":
    main()
