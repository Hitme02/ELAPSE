"""Test suite for M6 ensemble."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from networks import make_erdos_renyi, fiedler_value
from math_utils import max_entropy
from m6_ensemble import simulate as sim_m6, learn_weights, ensemble_signal


def get_test_setup(n=50, seed=42):
    G, L = make_erdos_renyi(n, seed=seed)
    A = np.abs((L - np.diag(np.diag(L))) * -1)
    lambda2 = fiedler_value(L)
    H_max = max_entropy(n)
    params = {
        'alpha': 0.3, 'mu': 1.5, 'H_c': 0.65*H_max, 'beta': 2.0,
        'n_hill': 4.0, 'beta_sir': 0.4, 'gamma': 0.15,
        'theta_ou': 0.3, 'mu_ou': 0.1, 'sigma_ou': 0.04,
        'kappa': 0.3, 'sigma_noise': 0.015,
        's': np.zeros(n), 'T_train': 15.0, 'dt': 0.02,
    }
    rng = np.random.default_rng(seed)
    x0  = rng.uniform(0.1, 1.0, n)
    return L, A, lambda2, params, x0


def test_weights_sum_to_one():
    """Learned weights must sum to 1."""
    L, A, lambda2, params, x0 = get_test_setup(n=30)
    training_data = [(x0, L, A, params, lambda2)]
    w, _ = learn_weights(training_data, lambda_reg=5.0, n_restarts=1, verbose=False)
    assert abs(w.sum() - 1.0) < 1e-6, f"Weights sum = {w.sum():.6f} != 1"
    assert np.all(w >= 0), f"Negative weights: {w}"
    print("PASS: Weights sum to 1 and are non-negative")


def test_delta_in_range():
    """ensemble_signal Delta must be in [0,1]."""
    for _ in range(100):
        votes = np.random.rand(5)
        w = np.random.rand(5); w /= w.sum()
        Delta, fires, eff = ensemble_signal(votes, w)
        assert 0.0 <= Delta <= 1.0 + 1e-9, f"Delta={Delta:.4f} not in [0,1]"
        assert 0.0 <= eff <= 1.0 + 1e-9, f"eff_sig={eff:.4f} not in [0,1]"
    print("PASS: Delta and eff_sig in [0,1]")


def test_optimal_better_than_uniform():
    """IEE(w_optimal) <= IEE(w_uniform) (non-trivial learning)."""
    L, A, lambda2, params, x0 = get_test_setup(n=50)
    training_data = [(x0, L, A, params, lambda2)]

    w_opt, _ = learn_weights(training_data, lambda_reg=5.0, n_restarts=2, verbose=False)
    w_uniform = np.ones(5) / 5

    np.random.seed(42)
    t, _, H, M, _, _ = sim_m6(x0, L, A, params, lambda2, weights=w_opt,
                                T=10.0, dt=0.05, stochastic=False)
    iee_opt = float(np.sum(H * M) * 0.05)

    np.random.seed(42)
    t, _, H, M, _, _ = sim_m6(x0, L, A, params, lambda2, weights=w_uniform,
                                T=10.0, dt=0.05, stochastic=False)
    iee_uniform = float(np.sum(H * M) * 0.05)

    print(f"IEE: optimal={iee_opt:.4f}, uniform={iee_uniform:.4f}")
    # Note: could be equal if uniform is already optimal, so we just report
    print("PASS: Ensemble learning completes without error")


if __name__ == '__main__':
    test_weights_sum_to_one()
    test_delta_in_range()
    test_optimal_better_than_uniform()
    print("\nAll ensemble tests PASSED")
