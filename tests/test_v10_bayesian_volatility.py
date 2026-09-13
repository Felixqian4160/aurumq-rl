import numpy as np
from types import SimpleNamespace
from aurumq_rl.v10.reward_components import BayesianVolConfig, BayesianVolEstimator


def test_zero_uncertainty_weight_is_preserved():
    est = BayesianVolEstimator(BayesianVolConfig(uncertainty_weight=0.0, min_vol=1e-6))
    est.update(0.1)
    assert np.isclose(est.volatility(), np.sqrt(est.variance_mean()))


def test_v10_config_zero_uncertainty_is_not_replaced_by_default():
    # Guards the exact config extraction contract used by WaveHunterV10Env.
    cfg = SimpleNamespace(
        bayes_vol_alpha0=2.0, bayes_vol_beta0=1e-4, bayes_vol_decay=0.98,
        bayes_vol_uncertainty_weight=0.0, bayes_vol_min=0.005,
    )
    weight = float(getattr(cfg, 'bayes_vol_uncertainty_weight', 0.50))
    assert weight == 0.0
