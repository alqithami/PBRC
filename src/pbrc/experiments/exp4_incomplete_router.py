
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from ..pbrc_core import Token, Message, Router, make_simple_binary_contract

def run(out_dir: Path, seed: int = 0) -> Path:
    rng = random.Random(seed)
    np.random.seed(seed)

    out_dir.mkdir(parents=True, exist_ok=True)

    # One-agent setting isolates router (in)completeness effects.
    b0 = np.array([0.1, 0.9], dtype=float)  # initially wrong if h0 is true
    true_h = "h0"
    evidence = Token("evidence_h0", supports=true_h)

    # Event always contains the same valid evidence token.
    event = [Message(sender="validator", text="verified tool output", tokens=(evidence,), confidence=1.0)]

    # Contract sets belief to correct distribution when evidence validates.
    contract = make_simple_binary_contract(h0="h0", h1="h1", p_set=0.9, fallback="identity")

    qs = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]
    R = 2000
    T_max = 30

    rows: List[Dict] = []
    for q in qs:
        router = Router(contract=contract, reject_empty_witness=True, p_false_negative=q, rng=rng)
        times: List[int] = []
        accepted: List[int] = []
        for r in range(R):
            b = b0.copy()
            t_hit = None
            for t in range(T_max):
                b, cert, acc = router.step(b, event)
                if acc:
                    accepted.append(1)
                else:
                    accepted.append(0)
                # correct if belief favors h0
                if int(np.argmax(b)) == 0:
                    t_hit = t + 1
                    break
            if t_hit is None:
                t_hit = T_max + 1
            times.append(t_hit)
        rows.append({
            "p_false_negative": q,
            "mean_time_to_correct": float(np.mean(times)),
            "median_time_to_correct": float(np.median(times)),
            "p_correct_within_Tmax": float(np.mean([t <= T_max for t in times])),
        })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "exp4_incomplete_router.csv"
    df.to_csv(csv_path, index=False)

    # Plot mean time to correct vs q
    fig_path = out_dir / "exp4_incomplete_router.png"
    plt.figure()
    plt.plot(df["p_false_negative"], df["mean_time_to_correct"], marker="o")
    plt.xlabel("Router false-negative rate (validation incompleteness)")
    plt.ylabel("Mean rounds to correct belief")
    plt.title("Sound-but-incomplete routers: safety preserved, liveness degrades")
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
