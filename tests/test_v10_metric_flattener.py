import math

from aurumq_rl.v10.metric_flattener import flatten_state


def test_flatten_state_with_nested_payload():
    state = {
        'hhi': 0.18,
        'effective_count': 6.0,
        'active_count': 8,
        'total_weight': 0.4,
        'nested': {'hhi': 0.99, 'should_ignore': True},
    }
    out = flatten_state(
        'hhi',
        state,
        allowed_keys=['hhi', 'effective_count', 'active_count', 'total_weight'],
    )
    assert out == {
        'hhi_hhi': 0.18,
        'hhi_effective_count': 6.0,
        'hhi_active_count': 8,
        'hhi_total_weight': 0.4,
    }


def test_flatten_state_skips_non_scalar_and_missing_keys():
    state = {'hhi': math.nan, 'extra': [1, 2, 3]}
    out = flatten_state(
        'hhi',
        state,
        allowed_keys=['hhi', 'effective_count'],
    )
    # NaN is not finite, so should be skipped.
    assert 'hhi_hhi' not in out
    assert 'hhi_effective_count' not in out


def test_flatten_state_returns_empty_when_input_is_none():
    assert flatten_state('cvar', None, ['a', 'b']) == {}
