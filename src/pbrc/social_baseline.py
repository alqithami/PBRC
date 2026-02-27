
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import networkx as nx

from .utils import normalize, log_pooling, sharpen

@dataclass(frozen=True)
class BaselineParams:
    """Baseline 'social influence' dynamics (no tokens)."""
    w0: float = 0.4       # base social weight
    ws: float = 0.5       # additional weight scaled by degree/(n-1)
    gamma: float = 2.0    # sharpening exponent (confidence amplification)
    T: int = 10           # rounds

def step_agent(b_i: np.ndarray, neigh_beliefs: np.ndarray, w_i: float, gamma: float) -> np.ndarray:
    s = neigh_beliefs.mean(axis=0)
    b = log_pooling(b_i, s, w_i)
    b = sharpen(b, gamma)
    return b

def simulate_trajectory(
    G: nx.Graph,
    init_beliefs: np.ndarray,
    params: BaselineParams
) -> List[np.ndarray]:
    """Return list of beliefs b^t, t=0..T for baseline social pooling."""
    n = G.number_of_nodes()
    b = init_beliefs.copy()
    neighbors = [list(G.neighbors(i)) + [i] for i in range(n)]  # include self
    degrees = np.array([G.degree(i) for i in range(n)], dtype=float)
    w_i = np.minimum(0.95, params.w0 + params.ws * (degrees / max(1.0, (n - 1.0))))

    traj = [b.copy()]
    for _ in range(params.T):
        new_b = np.zeros_like(b)
        for i in range(n):
            neigh_b = b[neighbors[i]]
            new_b[i] = step_agent(b[i], neigh_b, float(w_i[i]), params.gamma)
        b = new_b
        traj.append(b.copy())
    return traj

def simulate_final(G: nx.Graph, init_beliefs: np.ndarray, params: BaselineParams) -> np.ndarray:
    return simulate_trajectory(G, init_beliefs, params)[-1]

