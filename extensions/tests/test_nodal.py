import numpy as np
from extensions.system.ipr import LinearIPR
from extensions.system.vlp import CallableVLP
from extensions.system.nodal import solve_operating_point

# Hand-solvable system: choke q = C (WHP - P_down), tubing Pwf = WHP + D, reservoir q = PI (P_res - Pwf)
P_DOWN, D, PI, P_RES = 400.0, 2500.0, 10.0, 4000.0
VLP = CallableVLP(lambda q, whp, cond: np.broadcast_to(np.asarray(whp, dtype=float) + D, np.broadcast(q, whp).shape).ravel())
IPR = LinearIPR(PI, q_ref=PI * (P_RES - 3000.0), pwf_ref=3000.0)

def _choke(c):
    return lambda whp: c * np.maximum(np.asarray(whp, dtype=float) - P_DOWN, 0.0)

def test_matches_analytical_operating_point():
    c = 20.0
    whp_exact = (PI * (P_RES - D) + c * P_DOWN) / (c + PI)
    sol = solve_operating_point(_choke(c), VLP, IPR, P_DOWN, {})
    assert abs(sol['whp'] - whp_exact) < 0.5
    assert abs(sol['q'] - c * (whp_exact - P_DOWN)) < 10.0
    assert abs(sol['pwf'] - (sol['whp'] + D)) < 1e-6

def test_closing_choke_raises_whp_and_lowers_rate():
    small, large = (solve_operating_point(_choke(c), VLP, IPR, P_DOWN, {}) for c in (5.0, 50.0))
    assert small['whp'] > large['whp']
    assert small['q'] < large['q']

def test_no_flow_when_reservoir_cannot_reach_separator():
    weak = LinearIPR(PI, q_ref=PI * 100.0, pwf_ref=2800.0)       # P_res = 2900 < D + P_DOWN
    assert solve_operating_point(_choke(20.0), VLP, weak, P_DOWN, {}) is None
