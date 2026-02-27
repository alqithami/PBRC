
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import numpy as np

from .utils import argmax_index, max_confidence

def cascade_metrics(b_final: np.ndarray, true_h: int = 0, conf_thresh: float = 0.9) -> Dict[str, float]:
    """Compute summary metrics for a final belief matrix (n x m)."""
    argmax = np.argmax(b_final, axis=1)
    conf = np.max(b_final, axis=1)
    all_wrong = bool(np.all(argmax != true_h))
    all_correct = bool(np.all(argmax == true_h))
    return dict(
        all_wrong_sure=float(all_wrong and np.all(conf >= conf_thresh)),
        all_correct_sure=float(all_correct and np.all(conf >= conf_thresh)),
        mean_conf=float(conf.mean()),
        frac_wrong=float(np.mean(argmax != true_h)),
    )

