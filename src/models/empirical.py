import numpy as np

# Gilbert-type correlations: P = A * R^B * q / S^C
# P = upstream (wellhead) pressure, psig; R = GLR, scf/STB; q = liquid rate, STB/d; S = choke size, 64ths
CORRELATIONS = {
    'Gilbert':   (10.00, 0.546, 1.89),
    'Ros':       (17.40, 0.500, 2.00),
    'Baxendell': (9.56,  0.546, 1.93),
    'Achong':    (3.82,  0.650, 1.88),
}

def predict_empirical(name, p_psig, glr_scf_stb, choke_64ths):
    """Liquid rate (STB/d) from a Gilbert-type correlation. Valid for critical flow only."""
    a, b, c = CORRELATIONS[name]
    return p_psig * np.asarray(choke_64ths) ** c / (a * np.asarray(glr_scf_stb) ** b)

def choke_pct_to_64ths(choke_pct, s_max, n=1.0):
    """Equivalent bean size for a choke given as % open, with flow area proportional to (% open)^n."""
    return s_max * (np.asarray(choke_pct) / 100.0) ** (n / 2)

def fit_choke_curve(name, p_psig, glr_scf_stb, choke_pct, q_obs, n=None):
    """Fits the fully-open bean size S_max (64ths), and the area exponent n unless it is given, in log space.

    log q = log(P / (A R^B)) + C log S_max + (C n / 2) log(% / 100), which is linear in the unknowns.
    Returns (S_max, n).
    """
    a, b, c = CORRELATIONS[name]
    resid = np.log(np.asarray(q_obs)) - np.log(np.asarray(p_psig) / (a * np.asarray(glr_scf_stb) ** b))
    x = np.log(np.asarray(choke_pct) / 100.0)
    if n is not None:
        return float(np.exp(np.mean(resid - c * n / 2 * x) / c)), float(n)
    X = np.column_stack([np.ones_like(x), x])
    (k0, k1), *_ = np.linalg.lstsq(X, resid, rcond=None)
    return float(np.exp(k0 / c)), float(2 * k1 / c)
