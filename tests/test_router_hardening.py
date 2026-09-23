import networkx as nx
import numpy as np
import pytest
from pbrc.pbrc_core import Token, Message, Router, make_simple_binary_contract, flood_token_knowledge, time_to_global_coverage


def test_two_token_trigger_gets_sufficient_certificate():
    c = make_simple_binary_contract()
    c.triggers['sup_h0'] = lambda tokens: len(tokens) >= 2
    tokens = (Token('a', supports='h0'), Token('b', supports='h0'))
    _, (trigger, witness), accepted = Router(c).step(np.array([.1, .9]), [Message('s', '', tokens)])
    assert accepted and len(witness) == 2 and c.triggers[trigger](witness)


def test_invalid_extractor_cannot_execute_operator():
    c = make_simple_binary_contract(fallback='identity')
    c.triggers['sup_h0'] = lambda tokens: len(tokens) >= 2
    tokens = (Token('a', supports='h0'), Token('b', supports='h0'))
    c.witness = lambda *args: {tokens[0]}
    c.operators['sup_h0'] = lambda *args: pytest.fail('Invalid witness executed operator')
    initial = np.array([.1, .9])
    out, cert, accepted = Router(c).step(initial, [Message('s', '', tokens)])
    assert not accepted and cert == ('bot', set())
    np.testing.assert_array_equal(initial, out)


def test_fallback_is_called_once():
    c = make_simple_binary_contract()
    calls = []
    def fallback(b, event):
        calls.append(1)
        return b
    c.fallback = fallback
    assert not Router(c).step(np.array([.6, .4]), [])[2]
    assert len(calls) == 1


def test_missing_global_coverage_is_explicit():
    trace = flood_token_knowledge(nx.empty_graph(2), {0: {Token('x')}}, 3)
    assert time_to_global_coverage(trace) is None


def test_invalid_token_does_not_win_priority():
    c = make_simple_binary_contract()
    tokens = (Token('bad', supports='h0'), Token('good', supports='h1'))
    _, cert, accepted = Router(c, validate=lambda t: t.token_id == 'good').step(np.array([.6, .4]), [Message('s', '', tokens)])
    assert accepted and cert[0] == 'sup_h1'


def test_random_generators_are_not_shared():
    a, b = Router(make_simple_binary_contract()), Router(make_simple_binary_contract())
    assert a.rng is not b.rng
