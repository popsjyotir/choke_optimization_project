from typing import Protocol
import numpy as np
import pandas as pd

class IPR(Protocol):
    """Inflow performance: liquid rate the reservoir delivers at a bottomhole flowing pressure."""
    def rate(self, pwf) -> np.ndarray: ...

class LinearIPR:
    """Straight-line IPR anchored at a measured operating point: q = q_ref - PI * (Pwf - Pwf_ref).

    Anchoring avoids estimating reservoir pressure, which Volve does not report and which unconstrained
    fits put at implausible values for several wells. The implied p_res is only an extrapolation of the line.
    """
    def __init__(self, pi, q_ref, pwf_ref):
        self.pi, self.q_ref, self.pwf_ref = pi, q_ref, pwf_ref

    @property
    def p_res(self):
        return self.pwf_ref + self.q_ref / self.pi

    def rate(self, pwf):
        return self.q_ref - self.pi * (np.asarray(pwf, dtype=float) - self.pwf_ref)

def fit_pi(history: pd.DataFrame, window_days=30, min_rows=5, min_windows=3):
    """Productivity index (STB/d/psi) from day-to-day variation within short windows (fixed effects).

    Each window gets its own reservoir level, so slow reservoir-pressure changes do not bias the slope.
    Returns None when there is too little data or the slope has the wrong sign.
    """
    h = history.dropna(subset=['Pwf_psig'])
    if h.empty:
        return None
    window = (h.Date - h.Date.min()).dt.days // window_days
    counts = window.map(window.value_counts())
    h, window = h[counts >= min_rows], window[counts >= min_rows]
    if window.nunique() < min_windows:
        return None
    dm = h[['Pwf_psig', 'Q_liquid_STB']] - h.groupby(window)[['Pwf_psig', 'Q_liquid_STB']].transform('mean')
    pi = -(dm.Pwf_psig * dm.Q_liquid_STB).sum() / (dm.Pwf_psig ** 2).sum()
    return float(pi) if pi > 0 else None

def anchored_ipr(history: pd.DataFrame, q_now, pwf_now, **kwargs):
    """Linear IPR through today's operating point, with PI fitted on the well's history."""
    pi = fit_pi(history, **kwargs)
    return LinearIPR(pi, q_now, pwf_now) if pi is not None else None
