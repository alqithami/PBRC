
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run(out_dir: Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)

    # Model: an event contains N tokens, exactly one is "relevant" for the trigger.
    # Full validation checks all N tokens (cost = N).
    # Short-circuit validation checks tokens sequentially until a relevant one is found.
    # Under random token ordering, expected cost = (N+1)/2. Under adversarial ordering, cost = N.
    Ns = list(range(1, 501, 5))
    trials = 2000

    rows: List[Dict] = []
    for N in Ns:
        # Monte Carlo estimate of expected checks with random ordering
        # position of relevant token is uniform in {1,...,N}
        positions = rng.integers(1, N+1, size=trials)
        cost_short = float(positions.mean())
        rows.append({
            "N_tokens": N,
            "cost_full": float(N),
            "cost_short_random": cost_short,
            "cost_short_adversarial": float(N),
        })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "exp5_cost_dos.csv"
    df.to_csv(csv_path, index=False)

    plt.figure()
    plt.plot(df["N_tokens"], df["cost_full"], label="Full validation (always N)")
    plt.plot(df["N_tokens"], df["cost_short_random"], label="Short-circuit (random order)")
    plt.plot(df["N_tokens"], df["cost_short_adversarial"], linestyle="--", label="Short-circuit (adversarial order)")
    plt.xlabel("Number of tokens in event")
    plt.ylabel("Validation checks (proxy cost)")
    plt.title("DoS surface: token flooding vs validation cost")
    plt.legend(fontsize=8)
    fig_path = out_dir / "exp5_cost_dos.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    return csv_path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="results/sim")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(Path(args.out), seed=args.seed)
