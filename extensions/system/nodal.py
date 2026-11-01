import numpy as np
import pandas as pd
from src.models import pvt

def choke_rate_function(predict, state: dict, choke_pct):
    """Turns a choke model predict(df) into q(WHP) at a fixed choke opening and downstream pressure."""
    def q_of_whp(whp):
        whp = np.atleast_1d(np.asarray(whp, dtype=float))
        df = pd.DataFrame({**state, 'Choke_pct': choke_pct}, index=range(len(whp)))
        df['WHP_psig'] = whp
        df['Pressure_Ratio'] = (df.Pdown_psig + pvt.P_ATM_PSI) / (whp + pvt.P_ATM_PSI)
        df['dP_choke_psi'] = np.maximum(whp - df.Pdown_psig, 0.0)
        df['Flow_index'] = np.sqrt(df.dP_choke_psi) * choke_pct / 100
        return np.maximum(np.asarray(predict(df), dtype=float), 0.0)
    return q_of_whp

def solve_operating_point(q_choke, vlp, ipr, p_down, conditions, n_grid=400):
    """Finds the WHP at which choke, tubing and reservoir rates agree.

    Residual(WHP) = q_choke(WHP) - ipr.rate(vlp.pwf(q_choke(WHP), WHP)). It is negative at WHP = P_down (the choke
    passes nothing while the reservoir can still flow) and rises with WHP. The grid is evaluated in one batch
    and the first sign change is refined by linear interpolation, which also tolerates the step-shaped
    responses of tree models.

    Returns {'q', 'whp', 'pwf'} or None when the well cannot flow or no balance exists in the bracket.
    """
    p_res = getattr(ipr, 'p_res', None)
    dp_static = float(vlp.pwf(0.0, 0.0, conditions)[0])           # tubing loss at zero rate
    whp_high = (p_res - dp_static) if p_res is not None else p_down * 4
    if not np.isfinite(whp_high) or whp_high <= p_down + 1:
        return None

    whp = np.linspace(p_down + 0.1, whp_high, n_grid)
    q = q_choke(whp)
    pwf = vlp.pwf(q, whp, conditions)
    resid = q - ipr.rate(pwf)

    if resid[0] >= 0:
        return None                                                # reservoir cannot lift fluid to P_down
    up = np.flatnonzero(resid >= 0)
    if up.size == 0:
        return None
    i = up[0]
    w = resid[i - 1] / (resid[i - 1] - resid[i])
    whp_star = whp[i - 1] + w * (whp[i] - whp[i - 1])
    q_star = q[i - 1] + w * (q[i] - q[i - 1])
    return {'q': float(q_star), 'whp': float(whp_star), 'pwf': float(vlp.pwf(q_star, whp_star, conditions)[0])}
