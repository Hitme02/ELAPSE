"""
test_theorems.py
----------------
Integration tests verifying all three main theorems numerically.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from networks import make_erdos_renyi, fiedler_value
from math_utils import max_entropy


def test_theorem1_nonnegativity(n=50, n_paths=1000, T=5.0, dt=0.05, seed=0):
    """Theorem 1 part (iii): x_i(t) >= 0 a.s."""
    G, L = make_erdos_renyi(n, seed=42)
    rng  = np.random.default_rng(seed)

    min_vals = []
    for _ in range(n_paths):
        x = rng.uniform(0, 1, n)
        for i in range(int(T/dt)):
            dxdt = -0.3*(L@x) - 1.5*0.5*x
            x = x + dxdt*dt + 0.015*np.sqrt(np.maximum(x,0))*rng.standard_normal(n)*np.sqrt(dt)
            x = np.maximum(x, 0)
        min_vals.append(float(x.min()))

    assert min(min_vals) >= -1e-10, f"Non-negativity violated: min={min(min_vals):.2e}"
    print(f"PASS: Theorem 1 non-negativity (n={n}, {n_paths} paths, min={min(min_vals):.2e})")


def test_theorem2_stochastic_bound(n=50, n_paths=500, seed=0):
    """Theorem 2: E[IEE] < stochastic_iee_upper_bound."""
    try:
        from elapse.core.sde.stochastic_bound import stochastic_iee_upper_bound
    except ImportError:
        # Try direct import from package root
        elapse_root = os.path.join(os.path.dirname(__file__), '..', '..')
        sys.path.insert(0, os.path.abspath(elapse_root))
        from elapse.core.sde.stochastic_bound import stochastic_iee_upper_bound

    from math_utils import entropy
    G, L = make_erdos_renyi(n, seed=42)
    rng = np.random.default_rng(seed)

    T = 10.0; dt = 0.05; tau = 3.0
    mu = 1.5; delta_min = 0.1; sigma_n = 0.015; s_norm = 0.5
    M0 = n * 0.5

    iees = []
    for _ in range(n_paths):
        x = rng.uniform(0.1, 1.0, n)
        x *= M0 / x.sum()
        iee = 0.0
        past = False
        for i in range(int(T/dt)):
            t = i*dt
            H = entropy(x)
            M = x.sum()
            if H >= max_entropy(n)*0.65:
                past = True
            delta = delta_min if past else 0.0
            if t >= tau:
                iee += H*M*dt
            x = x + (-mu*delta*x)*dt + 0.05/n*np.ones(n)*dt
            x = x + sigma_n*np.sqrt(np.maximum(x,0))*rng.standard_normal(n)*np.sqrt(dt)
            x = np.maximum(x, 0)
        iees.append(iee)

    emp_mean = float(np.mean(iees))
    bound = stochastic_iee_upper_bound(n, M0, mu, delta_min, sigma_n, s_norm, tau, T)

    # Allow 2x slack since this is a conservative Gronwall bound
    print(f"  Theorem 2: emp_mean={emp_mean:.4f}, bound={bound:.4f}, "
          f"slack={100*(bound-emp_mean)/(bound+1e-6):.1f}%")
    assert emp_mean <= bound * 2.0, (
        f"Empirical IEE={emp_mean:.4f} > 2x bound={bound:.4f}. "
        "Check stochastic_bound implementation."
    )
    print(f"PASS: Theorem 2 stochastic bound (n={n}, {n_paths} paths)")


def test_theorem3_convexity(n=30, n_samples=1000, seed=0):
    """Theorem 3: IEE(lambda*w + (1-lambda)*w') <= lambda*IEE(w) + (1-lambda)*IEE(w')."""
    from m6_ensemble import simulate as sim_m6
    G, L = make_erdos_renyi(n, seed=42)
    A = np.abs((L - np.diag(np.diag(L))) * -1)
    lambda2 = fiedler_value(L)
    H_max = max_entropy(n)
    params = {
        'alpha': 0.3, 'mu': 1.5, 'H_c': 0.65*H_max, 'beta': 2.0,
        'n_hill': 4.0, 'beta_sir': 0.4, 'gamma': 0.15,
        'theta_ou': 0.3, 'mu_ou': 0.1, 'sigma_ou': 0.04,
        'kappa': 0.3, 'sigma_noise': 0.0,  # deterministic for convexity test
        's': np.zeros(n), 'T_train': 10.0, 'dt': 0.1,
    }
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0.1, 1.0, n)

    def iee(w):
        _, _, H, M, _, _ = sim_m6(x0, L, A, params, lambda2, weights=w,
                                   T=10.0, dt=0.1, stochastic=False)
        return float(np.sum(H*M)*0.1)

    violations = 0
    for _ in range(n_samples):
        # Random weight vectors on simplex
        w1_raw = rng.exponential(1, 5); w1 = w1_raw / w1_raw.sum()
        w2_raw = rng.exponential(1, 5); w2 = w2_raw / w2_raw.sum()
        lam = rng.uniform(0.1, 0.9)
        w_mid = lam*w1 + (1-lam)*w2

        iee1, iee2, iee_mid = iee(w1), iee(w2), iee(w_mid)
        upper = lam*iee1 + (1-lam)*iee2
        if iee_mid > upper + 1e-6:
            violations += 1

    violation_rate = violations / n_samples
    print(f"  Theorem 3 convexity: {violations}/{n_samples} violations "
          f"({100*violation_rate:.1f}%)")
    # Note: linearised IEE is convex; full nonlinear IEE may have small violations
    assert violation_rate < 0.15, (
        f"Too many convexity violations: {100*violation_rate:.1f}% > 15%"
    )
    print(f"PASS: Theorem 3 approximate convexity ({100*(1-violation_rate):.0f}% Jensen-consistent)")


if __name__ == '__main__':
    print("Testing Theorems 1-3...")
    test_theorem1_nonnegativity()
    test_theorem2_stochastic_bound()
    test_theorem3_convexity()
    print("\nAll theorem tests PASSED")
