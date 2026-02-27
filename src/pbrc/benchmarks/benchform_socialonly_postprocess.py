
"""
PBRC post-processor for BenchForm outputs (social_only mode).

Usage:
  python -m pbrc.benchmarks.benchform_socialonly_postprocess \
    --raw /path/to/raw.json \
    --social /path/to/trust_or_doubt.json \
    --out /path/to/pbrc_socialonly.json

Interpretation:
- `raw.json` is the subject model's output without peer guidance.
- `social.json` is the subject model's output with peer guidance (trust/doubt).

PBRC social_only enforcement:
- if social prediction differs from raw prediction, PBRC rejects the flip
  (no validated evidence tokens in the benchmark), so the enforced prediction
  is set back to the raw prediction.

This implements the evaluation methodology described in the paper's Section 13:
paired-run enforcement, producing auditable "no social-only flip" traces.

Note: This is a *conservative* enforcement variant. If you add tool-backed tokens,
extend this script to allow flips when a validated token is present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

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

    with open(raw_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    with open(soc_path, "r", encoding="utf-8") as f:
        soc = json.load(f)

    # BenchForm format: {config:..., outputs:{y_pred:[...], y_true:[...], ...}}
    raw_pred = raw["outputs"]["y_pred"]
    soc_pred = soc["outputs"]["y_pred"]

    assert len(raw_pred) == len(soc_pred), "raw/social length mismatch"

    enforced = soc.copy()
    flips = 0
    for i in range(len(soc_pred)):
        if soc_pred[i] != raw_pred[i]:
            enforced["outputs"]["y_pred"][i] = raw_pred[i]
            flips += 1

    enforced["pbrc"] = {
        "mode": "social_only",
        "flips_rejected": flips,
        "rule": "reject any y_pred flip relative to RAW (no evidence tokens)",
        "raw_source": str(raw_path),
        "social_source": str(soc_path),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(enforced, f, indent=2)

    print(f"Wrote {out_path}")
    print(f"Rejected flips: {flips}/{len(soc_pred)}")

if __name__ == "__main__":
    main()
