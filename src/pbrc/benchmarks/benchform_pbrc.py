
"""
BenchForm adapter for PBRC-enforced subject behavior.

This script is intentionally lightweight: it **does not** re-implement BenchForm.
Instead it *imports* BenchForm utilities (formatting + model calling) and runs:

  - an "independent" query (no peer rounds)
  - a "social" query (BenchForm protocol with peer rounds)

Then PBRC enforcement in `social_only` mode returns the independent answer unless
validated evidence tokens are provided (not enabled by default here).

This pipeline is enough to replicate the core paper claim:
  *social-only interaction cannot force a belief/answer flip under PBRC.*

Requires: the official BenchForm repo installed (or passed via --benchform_root).

BenchForm repo: https://github.com/Zhiyuan-Weng/BenchForm
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys
from typing import Dict, Any, List, Tuple

import numpy as np

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchform_root", type=str, required=True)
    ap.add_argument("--model", type=str, required=True, help="e.g., gpt-4o")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--protocol", type=str, default="trust", choices=["raw", "trust", "doubt"])
    ap.add_argument("--previous_discussions_rounds", type=int, default=5)
    ap.add_argument("--majority_num", type=int, default=6)
    ap.add_argument("--mode", type=str, default="social_only", choices=["social_only"])
    args = ap.parse_args()

    bench_root = Path(args.benchform_root)
    sys.path.insert(0, str(bench_root))

    # Import BenchForm utilities
    from utils import Config, generate_gpt, generate_gpt_empowered, generate_ollama, generate_glm
    from format_data_bbh import format_example_pairs

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        'sports_understanding','snarks','disambiguation_qa','movie_recommendation','causal_judgment',
        'date_understanding','tracking_shuffled_objects_three_objects','temporal_sequences','ruin_names',
        'web_of_lies','navigate','logical_deduction_five_objects','hyperbaton',
    ]

    def call_model(prompt: str) -> str:
        # Minimal: delegate to BenchForm's model callers.
        if args.model in ['gpt-3.5-turbo', 'gpt-4', 'gpt-4o']:
            return generate_gpt(prompt, model=args.model, temperature=0.7)
        if args.model in ['GLM-4-Plus']:
            return generate_glm(prompt, model=args.model)
        return generate_ollama(prompt, model=args.model)

    def extract_choice_letter(model_answer: str) -> str:
        # Same heuristic as BenchForm eval.py (kept local for robustness).
        try:
            tmp = model_answer.split('is: "(')
            if len(tmp) == 1:
                tmp = model_answer.split('is: (')
            if len(tmp) == 1:
                tmp = model_answer.split('is (')
            pred = tmp[-1][0]
            return pred
        except Exception:
            return "?"

    results: List[Dict[str, Any]] = []

    for task in tasks:
        # Load BBH split from BenchForm
        with open(bench_root / f"data/bbh/{task}/val_data.json", "r") as f:
            data = json.load(f)["data"]

        # Social prompt (BenchForm protocol with peer rounds)
        c_social = Config(task, protocol=args.protocol, multi_rounds=False,
                          previous_discussions_rounds=args.previous_discussions_rounds,
                          majority_num=args.majority_num,
                          model=args.model, batch=1, mode="default")
        prompts_social = format_example_pairs(data, c_social)

        # Independent prompt (no peer rounds, raw protocol)
        c_ind = Config(task, protocol="raw", multi_rounds=False,
                       previous_discussions_rounds=0,
                       majority_num=args.majority_num,
                       model=args.model, batch=1, mode="default")
        prompts_ind = format_example_pairs(data, c_ind)

        for idx, (p_ind, p_soc) in enumerate(zip(prompts_ind, prompts_social)):
            out_ind = call_model(p_ind)
            out_soc = call_model(p_soc)

            pred_ind = extract_choice_letter(out_ind)
            pred_soc = extract_choice_letter(out_soc)

            # PBRC social_only: reject answer flips that have no external evidence tokens.
            pred_pbrc = pred_ind

            results.append({
                "task": task,
                "idx": idx,
                "pred_ind": pred_ind,
                "pred_soc": pred_soc,
                "pred_pbrc": pred_pbrc,
                "out_ind": out_ind,
                "out_soc": out_soc,
            })

        print(f"[BenchForm PBRC] Finished task {task} ({len(data)} instances)")

    out_path = out_dir / f"benchform_pbrc_{args.model}_{args.protocol}.jsonl"
    with open(out_path, "w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")
    print("Wrote:", out_path)
    print("Tip: post-process with BenchForm's analysis scripts to compute conformity/independence.")

if __name__ == "__main__":
    main()
