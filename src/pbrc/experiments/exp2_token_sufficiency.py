
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple
import random

import numpy as np
import pandas as pd

from ..pbrc_core import Token, Message, PBRCContract, Router, make_simple_binary_contract, token_set

def random_event_with_tokens(toks: Set[Token], rhetoric_variant: int, rng: random.Random) -> List[Message]:
    msgs: List[Message] = []
    toks_list = list(toks)
    for idx, tok in enumerate(toks_list):
        msgs.append(
            Message(
                sender=f"a{(idx + rhetoric_variant) % 5}",
                text=f"Variant {rhetoric_variant}: token message {idx} (confidence={rng.random():.2f})",
                tokens=(tok,),
                confidence=rng.random(),
            )
        )
    # Add rhetoric-only messages (no tokens)
    for j in range(2):
        msgs.append(
            Message(
                sender=f"b{(j + rhetoric_variant) % 3}",
                text=f"Variant {rhetoric_variant}: rhetoric-only {j} majority says ...",
                tokens=(),
                confidence=rng.random(),
            )
        )
    rng.shuffle(msgs)
    return msgs

def run(out_dir: Path, seed: int = 0) -> Path:
    rng = random.Random(seed)
    np.random.seed(seed)

    contract = make_simple_binary_contract(fallback="identity")
    router = Router(contract=contract, reject_empty_witness=True, rng=rng)

    t_h0 = Token("t_h0", supports="h0")
    t_h1 = Token("t_h1", supports="h1")

    universe = [set(), {t_h0}, {t_h1}, {t_h0, t_h1}]

    N = 2000
    mismatches = 0
    accepts = 0
    rows: List[Dict] = []

    b0 = np.array([0.6, 0.4], dtype=float)

    for k in range(N):
        toks = rng.choice(universe)
        E1 = random_event_with_tokens(toks, rhetoric_variant=0, rng=rng)
        E2 = random_event_with_tokens(toks, rhetoric_variant=1, rng=rng)

        b1, cert1, acc1 = router.step(b0, E1)
        b2, cert2, acc2 = router.step(b0, E2)

        accepts += int(acc1)
        mismatches += int(not np.allclose(b1, b2))

        rows.append({
            "trial": k,
            "token_set": ",".join(sorted([t.token_id for t in toks])) if toks else "EMPTY",
            "accepted": int(acc1),
            "cert_trigger": cert1[0],
            "witness_size": len(cert1[1]),
            "equal_updates": int(np.allclose(b1, b2)),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = out_dir / "exp2_token_sufficiency.csv"
    df.to_csv(csv_path, index=False)

    # Write a tiny summary text file
    report = out_dir / "exp2_token_sufficiency_summary.txt"
    report.write_text(
        f"Trials: {N}\n"
        f"Accepted (nonempty-witness) steps: {accepts}\n"
        f"Mismatched updates across token-equivalent events: {mismatches}\n"
        f"Mismatch rate: {mismatches / N:.6f}\n",
        encoding="utf-8",
    )

    return csv_path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="results/sim")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(Path(args.out), seed=args.seed)
