
"""
KAIROS adapter for PBRC-enforced subject behavior.

KAIROS repo: https://github.com/declare-lab/KAIROS
Dataset: https://huggingface.co/datasets/declare-lab/KAIROS_EVAL

This adapter is a *thin wrapper*:
  - it loads KAIROS eval items (requires HF datasets)
  - runs subject model on a prompt WITHOUT peers (independent)
  - runs subject model on the full prompt WITH peers (social)
  - PBRC social_only returns the independent answer unless validated evidence is present

Because KAIROS supports multiple deployment modes (OpenAI, Azure, vLLM),
this script provides a generic hook `call_model(prompt)`. You should
edit that function to match your inference stack.

We do not bundle the KAIROS dataset here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import os
from typing import Any, Dict, List

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kairos_root", type=str, required=False, help="optional: path to local KAIROS repo")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--subject_model", type=str, required=True, help="OpenAI model name or your endpoint id")
    ap.add_argument("--mode", type=str, default="social_only", choices=["social_only"])
    ap.add_argument("--max_items", type=int, default=200)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except Exception as e:
        raise SystemExit(
            "This adapter requires `datasets` (HuggingFace). Install via `pip install datasets`."
        )

    # Load dataset from HF hub
    ds = load_dataset("declare-lab/KAIROS_EVAL", split="test")
    ds = ds.select(range(min(args.max_items, len(ds))))

    # ---- EDIT THIS FUNCTION for your inference stack ----
    def call_model(prompt: str) -> str:
        # Placeholder: implement OpenAI/vLLM call here.
        # Example (OpenAI python SDK v1):
        #   from openai import OpenAI
        #   client = OpenAI()
        #   resp = client.chat.completions.create(model=args.subject_model, messages=[{"role":"user","content":prompt}])
        #   return resp.choices[0].message.content
        raise NotImplementedError("Implement call_model(prompt) for your model stack.")

    results: List[Dict[str, Any]] = []

    for ex in ds:
        # Assumption: KAIROS items contain fields for (question, peers, etc).
        # If the schema differs, adjust prompt rendering accordingly.
        q = ex.get("question", "")
        peers = ex.get("peers", ex.get("peer_responses", ""))

        prompt_ind = f"Answer the question.\n\nQ: {q}\nA:"
        prompt_soc = f"Answer the question after considering peers.\n\nQ: {q}\n\nPeer responses:\n{peers}\n\nA:"

        out_ind = call_model(prompt_ind)
        out_soc = call_model(prompt_soc)

        # PBRC social_only: reject social-only revisions
        out_pbrc = out_ind

        results.append({
            "id": ex.get("id", None),
            "prompt_ind": prompt_ind,
            "prompt_soc": prompt_soc,
            "out_ind": out_ind,
            "out_soc": out_soc,
            "out_pbrc": out_pbrc,
        })

    out_path = out_dir / f"kairos_pbrc_{args.subject_model}.jsonl"
    with open(out_path, "w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")
    print("Wrote:", out_path)

if __name__ == "__main__":
    main()
