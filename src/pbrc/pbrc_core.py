
from __future__ import annotations

from dataclasses import dataclass, field
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
    """Contract over validated token sets. Theory-specific invariance is a caller obligation."""
    triggers: Dict[str, TriggerFn]
    operators: Dict[str, OperatorFn]
    priority: List[str]                 # first satisfied trigger in this order fires
    fallback: FallbackFn

    def __post_init__(self) -> None:
        names = set(self.triggers)
        if ("bot" in names or names != set(self.operators)
                or len(self.priority) != len(names) or set(self.priority) != names):
            raise ValueError("Triggers, operators and unique priorities must have identical keys")
        if not callable(self.fallback) or not all(
            callable(fn) for fn in [*self.triggers.values(), *self.operators.values()]
        ):
            raise TypeError("Contract components must be callable")

    def select_trigger(self, toks: Set[Token]) -> Optional[str]:
        satisfied = {name for name, fn in self.triggers.items() if fn(toks)}
        for name in self.priority:
            if name in satisfied:
                return name
        return None

    def witness(self, trig_name: Optional[str], toks: Set[Token]) -> Set[Token]:
        """Return a sufficient witness, not necessarily a minimum-cardinality one.

        A name is only a hint. A singleton must actually satisfy the predicate.
        For general set predicates use the full validated set when necessary.
        """
        if trig_name is None:
            return set()
        trigger = self.triggers[trig_name]
        for tok in sorted(toks, key=lambda value: (
            value.token_id, value.supports or "", value.contradicts or "",
            value.timestamp, value.attestations,
        )):
            if trigger({tok}):
                return {tok}
        return set(toks) if toks and trigger(toks) else set()

    def apply(self, b: np.ndarray, event: Event, toks: Set[Token]) -> Tuple[np.ndarray, Tuple[str, Set[Token]]]:
        """Execute exactly one operator, after checking witness sufficiency.

        ``toks`` must already be validated. The Router supplies this precondition.
        Triggers are pure predicates over token sets, not arbitrary first-order
        formulas over messages. Operator token-invariance is a caller assumption.
        """
        trig = self.select_trigger(toks)
        if trig is not None:
            W = set(self.witness(trig, toks))
            if W and W.issubset(toks) and self.triggers[trig](W):
                return self.operators[trig](b, event, toks), (trig, W)
        return self.fallback(b, event), ("bot", set())

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
    rng: random.Random = field(default_factory=lambda: random.Random(0))

    def __post_init__(self) -> None:
        if not self.reject_empty_witness:
            raise ValueError("The maintained Router always rejects empty witnesses")
        if not 0.0 <= self.p_false_negative <= 1.0:
            raise ValueError("p_false_negative must lie in [0, 1]")

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
        # apply() has already executed either the witnessed operator or fallback.
        # In particular, do not invoke a stochastic/stateful fallback twice.
        return b2, cert, trig != "bot" and bool(W)

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
    if T < 0 or set(G.nodes()) != set(range(n)):
        raise ValueError("T must be nonnegative and graph nodes must be 0..n-1")
    if not set(initial_tokens_by_node).issubset(G.nodes()):
        raise ValueError("Initial tokens refer to nodes absent from the graph")
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

def time_to_global_coverage(traces: List[List[Set[Token]]]) -> Optional[int]:
    """First observed closure time, or None when closure was never observed."""
    if not traces:
        raise ValueError("At least one time slice is required")
    if any(len(row) != len(traces[0]) for row in traces):
        raise ValueError("Node count changed across the exposure trace")
    all_tokens: Set[Token] = set()
    for tokset in traces[0]:
        all_tokens.update(tokset)
    for t, Kt in enumerate(traces):
        if all(all_tokens.issubset(Ki) for Ki in Kt):
            return t
    return None

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
    if h0 == h1 or not 0.0 <= p_set <= 1.0:
        raise ValueError("Hypotheses must differ and p_set must lie in [0, 1]")
    if fallback not in {"identity", "dilution"}:
        raise ValueError("fallback must be identity or dilution")
    if fallback == "dilution" and not 0.0 <= dilution_lam < 1.0:
        raise ValueError("Argmax-preserving dilution requires 0 <= lambda < 1")

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

