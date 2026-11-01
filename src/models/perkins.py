import numpy as np
from src.models import pvt

CD_DEFAULT = 0.826  # Perkins (1993) recommended discharge coefficient
R_GRID = np.linspace(0.01, 0.999, 1000)

def _mixture(p1_psia, t_f, gor, wc, fluid):
    """Per-STB-of-liquid masses and specific volumes at upstream conditions (frozen composition)."""
    api, gas_sg, water_sg, k = fluid['api'], fluid['gas_sg'], fluid['water_sg'], fluid['gas_k']
    rs = np.minimum(pvt.standing_rs(p1_psia, t_f, api, gas_sg), gor)
    bo = pvt.standing_bo(rs, t_f, api, gas_sg)
    rho_g1 = pvt.gas_density(p1_psia, t_f, gas_sg)

    gas_std_density = pvt.RHO_AIR_STD * gas_sg
    m_oil = (1 - wc) * (pvt.oil_sg(api) * pvt.RHO_WATER * pvt.STB_TO_M3 + rs * pvt.SCF_TO_M3 * gas_std_density)
    m_gas = (1 - wc) * (gor - rs) * pvt.SCF_TO_M3 * gas_std_density
    m_wat = wc * water_sg * pvt.RHO_WATER * pvt.STB_TO_M3
    m_tot = m_oil + m_gas + m_wat

    x_o, x_g, x_w = m_oil / m_tot, m_gas / m_tot, m_wat / m_tot
    v_o = np.where(m_oil > 0, (1 - wc) * bo * pvt.STB_TO_M3 / np.maximum(m_oil, 1e-12), 0.0)
    v_w = 1.0 / (water_sg * pvt.RHO_WATER)
    v_g1 = 1.0 / rho_g1

    # Polytropic exponent (Perkins 1993): n = 1 + x_g (Cp - Cv) / (x_g Cv + x_o Co + x_w Cw)
    r_gas = 8314.46 / (28.97 * gas_sg)          # Cp - Cv, J/(kg K)
    cv_g = r_gas / (k - 1)
    n = 1 + x_g * r_gas / (x_g * cv_g + x_o * 2000.0 + x_w * 4186.0)
    return dict(x_o=x_o, x_g=x_g, x_w=x_w, v_o=v_o, v_w=v_w, v_g1=v_g1, n=n, m_tot=m_tot)

def mass_flux(r, p1_pa, mix):
    """Throat mass flux (kg/s/m2) for downstream/upstream ratio r, neglecting upstream velocity."""
    x_o, x_g, x_w = mix['x_o'], mix['x_g'], mix['x_w']
    v_o, v_w, v_g1, n = mix['v_o'], mix['v_w'], mix['v_g1'], mix['n']
    v_liq = x_o * v_o + x_w * v_w
    # With no free gas (all gas in solution) n = 1 and the gas term is zero, not 0 * inf
    has_gas = x_g > 0
    n_safe = np.where(has_gas, n, 2.0)
    gas_term = np.where(has_gas, x_g * n_safe / (n_safe - 1) * v_g1 * (1 - r ** ((n_safe - 1) / n_safe)), 0.0)
    energy = gas_term + v_liq * (1 - r)
    v_m2 = x_g * v_g1 * r ** (-1 / n) + v_liq
    return np.sqrt(2 * p1_pa * np.maximum(energy, 0.0)) / v_m2

def critical_ratio(p1_pa, mix):
    """Pressure ratio that maximizes mass flux (critical flow boundary)."""
    cols = {key: np.atleast_1d(val)[:, None] for key, val in mix.items()}
    flux = mass_flux(R_GRID[None, :], np.atleast_1d(p1_pa)[:, None], cols)
    return R_GRID[np.argmax(flux, axis=1)]

def predict_perkins(p1_psig, p2_psig, t_f, gor, wc, choke_64ths, cd=CD_DEFAULT, fluid=None):
    """Liquid rate (STB/d) through a choke from the Perkins (1993) homogeneous multiphase model.

    Assumptions: homogeneous no-slip flow, frozen composition across the choke, polytropic gas
    expansion, incompressible liquids, friction lumped into cd, upstream velocity neglected.
    Returns (q_liquid_stb_d, critical_ratio, is_critical).
    """
    fluid = {**pvt.DEFAULT_FLUID, **(fluid or {})}
    p1, p2, t_f, gor, wc, choke = (np.atleast_1d(v) for v in np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (p1_psig, p2_psig, t_f, gor, wc, choke_64ths))))
    p1, p2 = p1 + pvt.P_ATM_PSI, p2 + pvt.P_ATM_PSI

    mix = _mixture(p1, t_f, gor, wc, fluid)
    p1_pa = p1 * pvt.PSI_TO_PA
    r_c = critical_ratio(p1_pa, mix)
    r_eff = np.maximum(p2 / p1, r_c)

    area = np.pi / 4 * (choke / 64 * 0.0254) ** 2
    w = cd * area * mass_flux(r_eff, p1_pa, mix)          # kg/s
    q_liquid = w / mix['m_tot'] * 86400.0                  # STB liquid per day
    return q_liquid, r_c, (p2 / p1) <= r_c

def fit_flow_factor(q_obs, q_pred):
    """Multiplier on Cd * S_max^2 that best matches observed rates in log space."""
    return float(np.exp(np.mean(np.log(q_obs / q_pred))))

def fit_choke_curve(q_obs, q_full_open, choke_pct):
    """Fits rate = factor * q_full_open * (% open / 100)^n by least squares in log space.

    q_full_open is Perkins at the fully-open reference bore. Perkins rate is proportional to flow area,
    so n is the exponent of the area-vs-opening curve and factor absorbs Cd and the true bore size.
    Returns (factor, n).
    """
    x = np.log(np.asarray(choke_pct) / 100.0)
    X = np.column_stack([np.ones_like(x), x])
    (k0, k1), *_ = np.linalg.lstsq(X, np.log(np.asarray(q_obs) / np.asarray(q_full_open)), rcond=None)
    return float(np.exp(k0)), float(k1)

def apply_choke_curve(q_full_open, choke_pct, factor, n):
    return factor * np.asarray(q_full_open) * (np.asarray(choke_pct) / 100.0) ** n
