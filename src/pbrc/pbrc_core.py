
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import random
import numpy as np
import networkx as nx

from .utils import normalize, skeptical_dilution

# -----------------------
# Tokens and events
# -----------------------

@dataclass(frozen=True)
class Token:
    """A minimal evidence token used in the simulations/adapters.

    In real deployments this would contain signatures, provenance, hashes, timestamps, etc.
    """
    token_id: str
    supports: Optional[str] = None       # hypothesis label supported
    contradicts: Optional[str] = None    # hypothesis label contradicted
    timestamp: int = 0
    attestations: int = 1

@dataclass(frozen=True)
class Message:
    sender: str
    text: str
    tokens: Tuple[Token, ...] = ()
    confidence: float = 0.0

Event = List[Message]

def token_set(event: Event) -> Set[Token]:
    toks: Set[Token] = set()
    for m in event:
        toks.update(m.tokens)
    return toks

# -----------------------
# Contracts
# -----------------------

TriggerFn = Callable[[Set[Token]], bool]
OperatorFn = Callable[[np.ndarray, Event, Set[Token]], np.ndarray]
FallbackFn = Callable[[np.ndarray, Event], np.ndarray]

@dataclass
class PBRCContract:
    """Token-invariant evidential contract used by the router."""
    triggers: Dict[str, TriggerFn]
    operators: Dict[str, OperatorFn]
    priority: List[str]                 # first satisfied trigger in this order fires
    fallback: FallbackFn

    def select_trigger(self, toks: Set[Token]) -> Optional[str]:
        satisfied = {name for name, fn in self.triggers.items() if fn(toks)}
        for name in self.priority:
            if name in satisfied:
                return name
        return None

    def witness(self, trig_name: Optional[str], toks: Set[Token]) -> Set[Token]:
        """Minimal witness: return one token sufficient for the trigger when possible."""
        if trig_name is None:
            return set()
        # common naming convention: "sup_<h>" or "con_<h>"
        if trig_name.startswith("sup_"):
            h = trig_name.split("_", 1)[1]
            for tok in toks:
                if tok.supports == h:
                    return {tok}
        if trig_name.startswith("con_"):
            h = trig_name.split("_", 1)[1]
            for tok in toks:
                if tok.contradicts == h:
                    return {tok}
        # fallback witness: all toks
        return set(toks)

    def apply(self, b: np.ndarray, event: Event, toks: Set[Token]) -> Tuple[np.ndarray, Tuple[str, Set[Token]]]:
        trig = self.select_trigger(toks)
        if trig is None:
            return self.fallback(b, event), ("bot", set())
        W = self.witness(trig, toks)
        if len(W) == 0:
            return self.fallback(b, event), ("bot", set())
        b2 = self.operators[trig](b, event, toks)
        return b2, (trig, W)

# -----------------------
# Routers
# -----------------------

ValidateFn = Callable[[Token], bool]

@dataclass
class Router:
    """State-holding router that enforces admissibility *and* operator compliance.

    This addresses the key enforceability boundary: the router computes b^{t+1} itself.
    """
    contract: PBRCContract
    validate: ValidateFn = lambda tok: True
    reject_empty_witness: bool = True

    # Optional incompleteness: false negative validation with probability p_fn
    p_false_negative: float = 0.0
    rng: random.Random = random.Random(0)

    def validate_tokens(self, toks: Set[Token]) -> Set[Token]:
        valid: Set[Token] = set()
        for tok in toks:
            if self.validate(tok):
                if self.p_false_negative > 0.0 and self.rng.random() < self.p_false_negative:
                    continue
                valid.add(tok)
        return valid

    def filter_event_to_valid_tokens(self, event: Event) -> Tuple[Event, Set[Token]]:
        toks = token_set(event)
        toks_valid = self.validate_tokens(toks)
        filtered: Event = []
        for m in event:
            filtered_tokens = tuple(tok for tok in m.tokens if tok in toks_valid)
            filtered.append(Message(sender=m.sender, text=m.text, tokens=filtered_tokens, confidence=m.confidence))
        return filtered, toks_valid

    def step(self, b: np.ndarray, event: Event) -> Tuple[np.ndarray, Tuple[str, Set[Token]], bool]:
        filtered_event, toks_valid = self.filter_event_to_valid_tokens(event)
        b2, cert = self.contract.apply(b, filtered_event, toks_valid)
        trig, W = cert
        if self.reject_empty_witness and (trig == "bot" or len(W) == 0):
            # gate rejects; enforce fallback only
            b_f = self.contract.fallback(b, filtered_event)
            return b_f, ("bot", set()), False
        return b2, cert, True

# -----------------------
# Dissemination (flooding)
# -----------------------

def flood_token_knowledge(
    G: nx.Graph,
    initial_tokens_by_node: Dict[int, Set[Token]],
    T: int
) -> List[List[Set[Token]]]:
    """Return K_i^t for each t=0..T (knowledge after t rounds of flooding)."""
    n = G.number_of_nodes()
    K: List[Set[Token]] = [set(initial_tokens_by_node.get(i, set())) for i in range(n)]
    traces: List[List[Set[Token]]] = []
    for _t in range(T + 1):
        traces.append([set(K[i]) for i in range(n)])
        # synchronous push flooding
        newK = [set(K[i]) for i in range(n)]
        for i in range(n):
            for j in G.neighbors(i):
                newK[j].update(K[i])
        K = newK
    return traces

def time_to_global_coverage(traces: List[List[Set[Token]]]) -> int:
    """Small helper: first t where all nodes have union of initial tokens."""
    all_tokens: Set[Token] = set()
    for tokset in traces[0]:
        all_tokens.update(tokset)
    for t, Kt in enumerate(traces):
        if all(all_tokens.issubset(Ki) for Ki in Kt):
            return t
    return len(traces) - 1

# -----------------------
# Common contract factories for experiments
# -----------------------

def make_simple_binary_contract(
    h0: str = "h0",
    h1: str = "h1",
    p_set: float = 0.9,
    fallback: str = "dilution",
    dilution_lam: float = 0.1,
) -> PBRCContract:
    """A 2-hypothesis token-invariant evidential PBRC contract used in simulations."""

    def trig_sup(h: str) -> TriggerFn:
        return lambda toks: any(tok.supports == h for tok in toks)

    def op_set(h: str) -> OperatorFn:
        def _op(b: np.ndarray, event: Event, toks: Set[Token]) -> np.ndarray:
            if h == h0:
                return np.array([p_set, 1.0 - p_set], dtype=float)
            return np.array([1.0 - p_set, p_set], dtype=float)
        return _op

    def fallback_fn(b: np.ndarray, event: Event) -> np.ndarray:
        if fallback == "identity":
            return b
        return skeptical_dilution(b, lam=dilution_lam)

    triggers = {
        f"sup_{h0}": trig_sup(h0),
        f"sup_{h1}": trig_sup(h1),
    }
    operators = {
        f"sup_{h0}": op_set(h0),
        f"sup_{h1}": op_set(h1),
    }
    priority = [f"sup_{h0}", f"sup_{h1}"]
    return PBRCContract(triggers=triggers, operators=operators, priority=priority, fallback=fallback_fn)

