
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from ..social_baseline import BaselineParams, simulate_final as simulate_baseline_final
from ..utils import skeptical_dilution
from ..metrics import cascade_metrics

def sample_init_beliefs(n: int, a: float = 5.0, b: float = 5.0) -> np.ndarray:
    """Binary beliefs [p(h0), p(h1)] sampled from a Beta distribution."""
    p = np.random.beta(a, b, size=n)
    return np.vstack([p, 1.0 - p]).T

def simulate_pbrc_social_only(init_b: np.ndarray, T: int = 10, lam: float = 0.1) -> np.ndarray:
    """No tokens => PBRC enforces fallback only. Here fallback = skeptical dilution."""
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

    # Baseline dynamics calibrated so dense graphs exhibit more wrong-but-sure cascades.
    base_params = BaselineParams(w0=0.4, ws=0.5, gamma=2.0, T=T)
    lam = 0.1
    a, b = 5.0, 5.0  # near-boundary: half runs start slightly on each side

    graphs = {
        "ring": nx.cycle_graph(n),
        "ER(p=0.3)": nx.erdos_renyi_graph(n, 0.3, seed=1),
        "complete": nx.complete_graph(n),
    }

    rows: List[Dict] = []
    for gname, G in graphs.items():
        for r in range(R):
            init_b = sample_init_beliefs(n, a=a, b=b)
            b_base = simulate_baseline_final(G, init_b, base_params)
            b_pbrc = simulate_pbrc_social_only(init_b, T=T, lam=lam)

            m_base = cascade_metrics(b_base, true_h=0, conf_thresh=conf_thresh)
            m_pbrc = cascade_metrics(b_pbrc, true_h=0, conf_thresh=conf_thresh)

            rows.append({"graph": gname, "model": "baseline", **m_base})
            rows.append({"graph": gname, "model": "PBRC (social-only fallback)", **m_pbrc})

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "exp1_social_cascades.csv"
    df.to_csv(csv_path, index=False)

    # Figure: wrong-but-sure cascade rate vs topology
    summary = df.groupby(["graph", "model"])["all_wrong_sure"].mean().reset_index()

    fig_path = out_dir / "exp1_social_cascades.png"
    plt.figure()
    for model in summary["model"].unique():
        sub = summary[summary["model"] == model]
        plt.plot(sub["graph"], sub["all_wrong_sure"], marker="o", label=model)
    plt.ylabel("Wrong-but-sure cascade rate")
    plt.xlabel("Topology")
    plt.title("Social-only wrong-but-sure cascades: baseline vs PBRC")
    plt.legend()
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
