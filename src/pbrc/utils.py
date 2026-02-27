
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set, Dict, Callable, Optional, Tuple
import numpy as np

EPS = 1e-12

def normalize(p: np.ndarray) -> np.ndarray:
    """Normalize a nonnegative vector into a probability simplex element."""
    p = np.asarray(p, dtype=float)
    p = np.clip(p, EPS, None)
    return p / p.sum()

def sharpen(p: np.ndarray, gamma: float) -> np.ndarray:
    """Sharpen a distribution (gamma>1 increases confidence)."""
    return normalize(np.power(p, gamma))

def skeptical_dilution(p: np.ndarray, lam: float = 0.1) -> np.ndarray:
    """Move distribution towards uniform (argmax-preserving if lam small and margin exists)."""
    p = np.asarray(p, dtype=float)
    m = p.shape[-1]
    u = np.ones(m, dtype=float) / m
    return (1.0 - lam) * p + lam * u

def log_pooling(b: np.ndarray, s: np.ndarray, w: float) -> np.ndarray:
    """Logarithmic opinion pooling."""
    b = np.asarray(b, dtype=float)
    s = np.asarray(s, dtype=float)
    return normalize(np.power(b, 1.0 - w) * np.power(s, w))

def argmax_index(p: np.ndarray) -> int:
    return int(np.argmax(p))

def max_confidence(p: np.ndarray) -> float:
    return float(np.max(p))

