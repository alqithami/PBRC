
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from ..pbrc_core import Token, flood_token_knowledge, time_to_global_coverage

def make_graphs(n: int = 20) -> Dict[str, nx.Graph]:
    graphs: Dict[str, nx.Graph] = {}
    graphs["ring"] = nx.cycle_graph(n)
    graphs["ER(p=0.2)"] = nx.erdos_renyi_graph(n, 0.2, seed=2)
    graphs["star"] = nx.star_graph(n - 1)
    graphs["complete"] = nx.complete_graph(n)
    # a 4x5 grid has 20 nodes
    graphs["grid(4x5)"] = nx.grid_2d_graph(4, 5)
    # relabel grid nodes to 0..n-1
    graphs["grid(4x5)"] = nx.convert_node_labels_to_integers(graphs["grid(4x5)"])
    return graphs

def run(out_dir: Path, seed: int = 0) -> Path:
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 20
    graphs = make_graphs(n)

    rows: List[Dict] = []
    for gname, G in graphs.items():
        if not nx.is_connected(G):
            continue

        # Unique token at each node => time to global coverage should equal diameter
        init_tokens: Dict[int, Set[Token]] = {
            i: {Token(f"tau_{i}", supports="h0" if i % 2 == 0 else "h1")} for i in range(G.number_of_nodes())
        }

        D = nx.diameter(G)
        T = D + 2  # simulate a bit longer than diameter
        traces = flood_token_knowledge(G, init_tokens, T=T)
        t_cov = time_to_global_coverage(traces)

        rows.append({
            "graph": gname,
            "n": G.number_of_nodes(),
            "m_edges": G.number_of_edges(),
            "diameter": D,
            "observed_coverage_time": t_cov,
            "matches": int(D == t_cov),
        })

    df = pd.DataFrame(rows)
    csv_path = out_dir / "exp3_topology_token_flow.csv"
    df.to_csv(csv_path, index=False)

    # Figure: observed coverage time vs diameter
    fig_path = out_dir / "exp3_topology_token_flow.png"
    plt.figure()
    plt.scatter(df["diameter"], df["observed_coverage_time"])
    for _, r in df.iterrows():
        plt.annotate(r["graph"], (r["diameter"], r["observed_coverage_time"]), fontsize=7)
    plt.plot([df["diameter"].min(), df["diameter"].max()],
             [df["diameter"].min(), df["diameter"].max()], linestyle="--")
    plt.xlabel("Graph diameter D(G)")
    plt.ylabel("Observed flooding time to global token coverage")
    plt.title("Flooding dissemination: closure time equals diameter (unique token per node)")
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
