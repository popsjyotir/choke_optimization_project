from typing import Callable, Protocol
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

class VLP(Protocol):
    """Vertical lift performance: bottomhole flowing pressure needed to lift a liquid rate to a given WHP."""
    def pwf(self, q_liquid, whp, conditions) -> np.ndarray: ...

def _features(q_liquid, water_cut, glr, whp):
    q = np.asarray(q_liquid, dtype=float) / 1e4
    q, water_cut, glr, whp = np.broadcast_arrays(q, water_cut, glr, whp)
    return np.column_stack([np.ones(q.size), water_cut.ravel(), glr.ravel() / 1000, whp.ravel() / 1000, q.ravel(), q.ravel() ** 2])

class DataVLP:
    """Tubing pressure loss fitted to a well's measured Pwf - WHP.

    dP_tubing = c0 + c1*WC + c2*GLR + c3*WHP + c4*q + c5*q^2: hydrostatic terms (water cut, gas, gas compression)
    plus a friction-like rate term. Valid within the rates and water cuts the well has produced at.
    """
    def __init__(self, coef, n_rows):
        self.coef, self.n_rows = np.asarray(coef), n_rows

    @classmethod
    def fit(cls, history: pd.DataFrame, min_rows=30):
        h = history.dropna(subset=['Pwf_psig'])
        if len(h) < min_rows:
            return None
        X = _features(h.Q_liquid_STB, h.Water_Cut, h.GLR_scf_stb, h.WHP_psig)
        coef, *_ = np.linalg.lstsq(X, h.DP_tubing_psi.values, rcond=None)
        return cls(coef, len(h))

    def pwf(self, q_liquid, whp, conditions):
        whp = np.asarray(whp, dtype=float)
        dp = _features(q_liquid, conditions['Water_Cut'], conditions['GLR_scf_stb'], whp) @ self.coef
        return np.broadcast_to(whp, np.broadcast(np.asarray(q_liquid), whp).shape).ravel() + dp

class TableVLP:
    """VLP from an exported table (e.g. PROSPER or PIPESIM), interpolated linearly.

    The CSV needs columns Q_liquid_STB, WHP_psig and Pwf_psig, and may add Water_Cut and GLR_scf_stb.
    Points outside the table use the nearest table value.
    """
    AXES = ['Q_liquid_STB', 'WHP_psig', 'Water_Cut', 'GLR_scf_stb']

    def __init__(self, table: pd.DataFrame):
        self.axes = [a for a in self.AXES if a in table.columns]
        pts = table[self.axes].values.astype(float)
        self.scale = pts.max(axis=0) - pts.min(axis=0)
        self.scale[self.scale == 0] = 1.0
        self.linear = LinearNDInterpolator(pts / self.scale, table.Pwf_psig.values)
        self.nearest = NearestNDInterpolator(pts / self.scale, table.Pwf_psig.values)

    @classmethod
    def from_csv(cls, path):
        return cls(pd.read_csv(path))

    def pwf(self, q_liquid, whp, conditions):
        values = {'Q_liquid_STB': q_liquid, 'WHP_psig': whp,
                  'Water_Cut': conditions.get('Water_Cut', 0.0), 'GLR_scf_stb': conditions.get('GLR_scf_stb', 0.0)}
        cols = np.broadcast_arrays(*(np.asarray(values[a], dtype=float) for a in self.axes))
        pts = np.column_stack([c.ravel() for c in cols]) / self.scale
        out = self.linear(pts)
        return np.where(np.isnan(out), self.nearest(pts), out)

class CallableVLP:
    """Wraps any function f(q_liquid, whp, conditions) -> pwf, e.g. your own tubing correlation."""
    def __init__(self, fn: Callable):
        self.fn = fn

    def pwf(self, q_liquid, whp, conditions):
        return np.asarray(self.fn(q_liquid, whp, conditions), dtype=float)
