
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from ..social_baseline import BaselineParams, simulate_final as simulate_baseline_final
from ..utils import skeptical_dilution
from ..metrics import cascade_metrics

def sample_init_beliefs(n: int, a: float = 5.0, b: float = 5.0) -> np.ndarray:
    p = np.random.beta(a, b, size=n)
    return np.vstack([p, 1.0 - p]).T

def simulate_pbrc_social_only(init_b: np.ndarray, T: int = 10, lam: float = 0.1) -> np.ndarray:
    b = init_b.copy()
    for _ in range(T):
        b = np.array([skeptical_dilution(bi, lam=lam) for bi in b])
    return b

def run(out_dir: Path, seed: int = 0) -> Path:
    np.random.seed(seed)

    n = 20
    T = 10
    R = 500
    conf_thresh = 0.9

    base_params = BaselineParams(w0=0.4, ws=0.5, gamma=2.0, T=T)
    a, b = 5.0, 5.0

    lambdas = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]

    graphs = {
        "ring": nx.cycle_graph(n),
        "ER(p=0.3)": nx.erdos_renyi_graph(n, 0.3, seed=1),
        "complete": nx.complete_graph(n),
    }

    # Precompute baseline once per graph (lambda-independent)
    baseline_rows: List[Dict] = []
    for gname, G in graphs.items():
        for r in range(R):
            init_b = sample_init_beliefs(n, a=a, b=b)
            b_base = simulate_baseline_final(G, init_b, base_params)
            m_base = cascade_metrics(b_base, true_h=0, conf_thresh=conf_thresh)
            baseline_rows.append({"graph": gname, "model": "baseline", "lambda": np.nan, **m_base})

    # PBRC rows per lambda
    rows: List[Dict] = []
    for lam in lambdas:
        for gname, G in graphs.items():
            for r in range(R):
                init_b = sample_init_beliefs(n, a=a, b=b)
                b_pbrc = simulate_pbrc_social_only(init_b, T=T, lam=lam)
                m_pbrc = cascade_metrics(b_pbrc, true_h=0, conf_thresh=conf_thresh)
                rows.append({"graph": gname, "model": "PBRC", "lambda": lam, **m_pbrc})

    df = pd.DataFrame(baseline_rows + rows)
    csv_path = out_dir / "exp1c_ablation_lambda.csv"
    df.to_csv(csv_path, index=False)

    # Plot cascade rate vs lambda for PBRC, with baseline as horizontal reference
    plt.figure()
    for gname in graphs.keys():
        sub = df[(df["graph"] == gname) & (df["model"] == "PBRC")]
        # average across trials at each lambda
        g = sub.groupby("lambda")["all_wrong_sure"].mean().reset_index()
        plt.plot(g["lambda"], g["all_wrong_sure"], marker="o", label=f"{gname} (PBRC)")
        # baseline horizontal
        base_rate = df[(df["graph"] == gname) & (df["model"] == "baseline")]["all_wrong_sure"].mean()
        plt.axhline(base_rate, linestyle="--", linewidth=1.0)

    plt.xlabel("Dilution parameter $\\lambda$")
    plt.ylabel("Wrong-but-sure cascade rate")
    plt.title("PBRC ablation: cascade rate vs dilution")
    plt.legend(fontsize=8)
    fig_path = out_dir / "exp1c_ablation_lambda.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    # Plot mean confidence vs lambda
    plt.figure()
    for gname in graphs.keys():
        sub = df[(df["graph"] == gname) & (df["model"] == "PBRC")]
        g = sub.groupby("lambda")["mean_conf"].mean().reset_index()
        plt.plot(g["lambda"], g["mean_conf"], marker="o", label=f"{gname} (PBRC)")
        base_conf = df[(df["graph"] == gname) & (df["model"] == "baseline")]["mean_conf"].mean()
        plt.axhline(base_conf, linestyle="--", linewidth=1.0)
    plt.xlabel("Dilution parameter $\\lambda$")
    plt.ylabel("Mean confidence at $T$")
    plt.title("PBRC ablation: confidence vs dilution")
    plt.legend(fontsize=8)
    fig2_path = out_dir / "exp1c_ablation_conf.png"
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=200)
    plt.close()

    return csv_path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="results/sim")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(Path(args.out), seed=args.seed)
