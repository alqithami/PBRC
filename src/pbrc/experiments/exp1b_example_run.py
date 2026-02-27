
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from ..social_baseline import BaselineParams, simulate_trajectory as baseline_traj
from ..utils import skeptical_dilution

def pbrc_traj(init_b: np.ndarray, T: int = 10, lam: float = 0.1) -> List[np.ndarray]:
    b = init_b.copy()
    traj = [b.copy()]
    for _ in range(T):
        b = np.array([skeptical_dilution(bi, lam=lam) for bi in b])
        traj.append(b.copy())
    return traj

def run(out_dir: Path, seed: int = 0) -> None:
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 20
    T = 10
    G = nx.complete_graph(n)

    # Initial condition: majority mildly correct, minority confidently wrong.
    init_p = np.zeros(n)
    idx = np.arange(n)
    np.random.shuffle(idx)
    mild = idx[: int(0.75 * n)]
    wrong = idx[int(0.75 * n):]
    init_p[mild] = np.random.uniform(0.55, 0.65, size=len(mild))
    init_p[wrong] = np.random.uniform(0.01, 0.05, size=len(wrong))
    init_b = np.vstack([init_p, 1.0 - init_p]).T

    base_params = BaselineParams(w0=0.4, ws=0.5, gamma=2.0, T=T)
    traj_base = baseline_traj(G, init_b, base_params)
    traj_pbrc = pbrc_traj(init_b, T=T, lam=0.1)

    def series(traj: List[np.ndarray]) -> pd.DataFrame:
        p_true = [float(bt[:, 0].mean()) for bt in traj]
        conf = [float(np.max(bt, axis=1).mean()) for bt in traj]
        frac_wrong = [float(np.mean(np.argmax(bt, axis=1) != 0)) for bt in traj]
        return pd.DataFrame({"t": np.arange(len(traj)), "mean_p_true": p_true, "mean_conf": conf, "frac_wrong": frac_wrong})

    df_base = series(traj_base); df_base["model"] = "baseline"
    df_pbrc = series(traj_pbrc); df_pbrc["model"] = "PBRC (fallback)"
    df = pd.concat([df_base, df_pbrc], ignore_index=True)
    df.to_csv(out_dir / "exp1b_example_run.csv", index=False)

    # Plot mean belief in true hypothesis
    plt.figure()
    for model, sub in df.groupby("model"):
        plt.plot(sub["t"], sub["mean_p_true"], marker="o", label=model)
    plt.xlabel("Round t")
    plt.ylabel("Mean belief in true hypothesis")
    plt.title("Representative run (complete graph): mean belief trajectory")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "exp1b_example_mean_ptrue.png", dpi=200)
    plt.close()

    # Plot mean confidence
    plt.figure()
    for model, sub in df.groupby("model"):
        plt.plot(sub["t"], sub["mean_conf"], marker="o", label=model)
    plt.xlabel("Round t")
    plt.ylabel("Mean confidence")
    plt.title("Representative run (complete graph): confidence amplification vs PBRC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "exp1b_example_mean_conf.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="results/sim")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(Path(args.out), seed=args.seed)
