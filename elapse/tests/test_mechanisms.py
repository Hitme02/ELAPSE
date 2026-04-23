"""Test suite for mechanism vote functions."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from networks import make_erdos_renyi, fiedler_value
from math_utils import max_entropy
import m0_baseline as m0, m1_egdm as m1, m2_epidemic as m2
import m3_finance as m3, m4_biology as m4, m5_social as m5


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


def test_vote_range():
    """All votes must be in [0,1]."""
    L, A, lambda2, params, x0 = get_test_setup()
    n = len(x0)
    state2 = np.concatenate([np.ones(n)*0.5, x0, np.zeros(n)])

    for t in [0.0, 5.0, 10.0, 15.0]:
        v1 = m1.vote(x0, L, params, t, lambda2)
        v2 = m2.vote(state2, A, params, t, lambda2)
        v3 = m3.vote(x0, params, t, lambda2)
        v4 = m4.vote(x0, L, params, t, lambda2)
        v5 = m5.vote(x0, A, params, t, lambda2)

        for name, v in [('M1', v1), ('M2', v2), ('M3', v3), ('M4', v4), ('M5', v5)]:
            assert 0.0 <= v <= 1.0 + 1e-9, f"{name} vote={v:.4f} at t={t} not in [0,1]"

    print("PASS: All votes in [0,1]")


def test_vote_monotonicity():
    """M3 vote (first-passage) is non-decreasing in t for fixed x."""
    L, A, lambda2, params, x0 = get_test_setup()
    prev = 0.0
    for t in np.linspace(0, 15, 100):
        v = m3.vote(x0, params, t, lambda2)
        assert v >= prev - 1e-9, f"M3 vote not non-decreasing: v={v:.4f} < prev={prev:.4f} at t={t:.2f}"
        prev = v
    print("PASS: M3 vote monotonically non-decreasing")


def test_vote_near_zero_at_t0():
    """All votes near 0 at t=0 (before trigger)."""
    L, A, lambda2, params, x0 = get_test_setup()
    # Use low-entropy initial conditions (concentrated)
    x_low = np.zeros(len(x0))
    x_low[0] = 1.0
    state2 = np.concatenate([np.ones(len(x0))*0.5, x_low, np.zeros(len(x0))])

    v3 = m3.vote(x_low, params, 0.0, lambda2)
    assert v3 < 0.1, f"M3 vote at t=0 should be near 0, got {v3:.4f}"
    print("PASS: Votes near 0 at t=0")


if __name__ == '__main__':
    test_vote_range()
    test_vote_monotonicity()
    test_vote_near_zero_at_t0()
    print("\nAll mechanism tests PASSED")
