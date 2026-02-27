"""
PBRC post-processor for KAIROS outputs (social_only mode).

Usage:
  python -m pbrc.benchmarks.kairos_socialonly_postprocess \
    --raw /path/to/RAW.json \
    --social /path/to/PROTOCOL.json \
    --out /path/to/PBRC_socialonly.json

PBRC social_only enforcement (post-hoc):
- reject any prediction flip relative to RAW when no validated evidence tokens are present.

This script supports both:
- standard outputs:        outputs["y_pred"]
- reflection outputs:      outputs["y_pred_reflected"] (if present)

It will enforce:
- y_pred_reflected flips if social has it (baseline: raw y_pred_reflected if present, otherwise raw y_pred)
- y_pred flips if present in both raw & social

It also updates is_correct / is_correct_reflected when y_true exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple


def _enforce_vector(
    raw_pred: List[Any],
    soc_pred: List[Any],
) -> Tuple[List[Any], int]:
    """Return enforced prediction list + number of flips rejected."""
    if len(raw_pred) != len(soc_pred):
        raise ValueError(f"raw/social length mismatch: {len(raw_pred)} vs {len(soc_pred)}")
    out = list(soc_pred)
    flips = 0
    for i, (r, s) in enumerate(zip(raw_pred, soc_pred)):
        if s != r:
            out[i] = r
            flips += 1
    return out, flips


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=str, required=True)
    ap.add_argument("--social", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    raw_path = Path(args.raw)
    soc_path = Path(args.social)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    soc = json.loads(soc_path.read_text(encoding="utf-8"))

    if "outputs" not in raw or "outputs" not in soc:
        raise KeyError("Both RAW and SOCIAL JSON must contain top-level key: outputs")

    raw_out: Dict[str, Any] = raw["outputs"]
    soc_out: Dict[str, Any] = soc["outputs"]

    enforced = soc.copy()
    enforced_out = enforced["outputs"]

    flips_rejected: Dict[str, int] = {}

    # 1) Enforce reflected predictions if SOCIAL has them.
    # Baseline preference: RAW y_pred_reflected if present; otherwise RAW y_pred.
    if "y_pred_reflected" in soc_out:
        if "y_pred_reflected" in raw_out:
            raw_base = raw_out["y_pred_reflected"]
        elif "y_pred" in raw_out:
            raw_base = raw_out["y_pred"]
        else:
            raise KeyError("RAW has neither y_pred_reflected nor y_pred; cannot enforce reflection outputs")

        enforced_vec, flips = _enforce_vector(raw_base, soc_out["y_pred_reflected"])
        enforced_out["y_pred_reflected"] = enforced_vec
        flips_rejected["y_pred_reflected"] = flips

    # 2) Enforce standard predictions if both RAW and SOCIAL have them.
    if "y_pred" in raw_out and "y_pred" in soc_out:
        enforced_vec, flips = _enforce_vector(raw_out["y_pred"], soc_out["y_pred"])
        enforced_out["y_pred"] = enforced_vec
        flips_rejected["y_pred"] = flips

    if not flips_rejected:
        raise KeyError("No enforceable prediction keys found. Expected y_pred and/or y_pred_reflected in SOCIAL.")

    # Update correctness arrays if y_true present
    y_true = enforced_out.get("y_true")
    if y_true is not None:
        if "y_pred" in enforced_out:
            enforced_out["is_correct"] = [int(p == t) for p, t in zip(enforced_out["y_pred"], y_true)]
        if "y_pred_reflected" in enforced_out:
            enforced_out["is_correct_reflected"] = [int(p == t) for p, t in zip(enforced_out["y_pred_reflected"], y_true)]

    enforced["pbrc"] = {
        "mode": "social_only",
        "flips_rejected": flips_rejected,
        "rule": "reject any prediction flip relative to RAW (no evidence tokens)",
        "raw_source": str(raw_path),
        "social_source": str(soc_path),
    }

    out_path.write_text(json.dumps(enforced, indent=2), encoding="utf-8")

    total_n = None
    if "y_pred_reflected" in enforced_out:
        total_n = len(enforced_out["y_pred_reflected"])
    elif "y_pred" in enforced_out:
        total_n = len(enforced_out["y_pred"])

    print(f"Wrote {out_path}")
    print("Rejected flips:", flips_rejected, f"(n={total_n})")


if __name__ == "__main__":
    main()
