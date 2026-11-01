import numpy as np
from src.models import perkins, pvt

def _gas_only_mix(k=1.3):
    gas_sg = 0.75
    r_gas = 8314.46 / (28.97 * gas_sg)
    return dict(x_o=np.array([0.0]), x_g=np.array([1.0]), x_w=np.array([0.0]),
                v_o=np.array([0.0]), v_w=np.array([0.0]), v_g1=np.array([0.01]),
                n=np.array([1 + r_gas / (r_gas / (k - 1))]), m_tot=np.array([1.0]))

def test_gas_only_critical_ratio_matches_isentropic_nozzle():
    k = 1.3
    r_c = perkins.critical_ratio(np.array([1e7]), _gas_only_mix(k))[0]
    assert abs(r_c - (2 / (k + 1)) ** (k / (k - 1))) < 0.002

def test_liquid_only_matches_bernoulli():
    mix = dict(x_o=np.array([0.0]), x_g=np.array([0.0]), x_w=np.array([1.0]), v_o=np.array([0.0]),
               v_w=np.array([0.001]), v_g1=np.array([1.0]), n=np.array([1.3]), m_tot=np.array([1.0]))
    flux = perkins.mass_flux(0.5, 1e6, mix)[0]
    assert abs(flux - np.sqrt(2 * 1000.0 * 0.5e6)) / flux < 1e-9

def test_rate_is_flat_below_critical_ratio_and_continuous():
    p1 = 1000.0
    ratios = np.linspace(0.2, 0.98, 80)
    q, r_c, _ = perkins.predict_perkins(p1, (p1 + pvt.P_ATM_PSI) * ratios - pvt.P_ATM_PSI, 150.0, 800.0, 0.3, 40.0)
    critical = ratios <= r_c[0]
    assert np.allclose(q[critical], q[critical][0])
    assert np.all(np.diff(q) <= 1e-6 * q.max())          # rate never increases as backpressure rises

    # No jump at the regime boundary: just above r_c the rate stays within 1% of the critical rate
    p2_above = (p1 + pvt.P_ATM_PSI) * (r_c[0] + 0.005) - pvt.P_ATM_PSI
    q_above, _, _ = perkins.predict_perkins(p1, p2_above, 150.0, 800.0, 0.3, 40.0)
    assert abs(q_above[0] - q[critical][0]) < 0.01 * q[critical][0]

def test_rate_goes_to_zero_as_pressure_drop_vanishes():
    q_near, _, _ = perkins.predict_perkins(1000.0, 999.5, 150.0, 800.0, 0.3, 40.0)
    q_open, _, _ = perkins.predict_perkins(1000.0, 200.0, 150.0, 800.0, 0.3, 40.0)
    assert q_near[0] < 0.05 * q_open[0]
