import numpy as np

# Default fluid properties (assumed; no Volve PVT report in this project)
DEFAULT_FLUID = {
    'api': 28.0,        # stock-tank oil gravity (deg API)
    'gas_sg': 0.75,     # gas specific gravity (air = 1)
    'water_sg': 1.02,   # produced water specific gravity
    'gas_k': 1.30,      # gas heat capacity ratio Cp/Cv
}

P_ATM_PSI = 14.696
RHO_WATER = 999.0        # kg/m3 at standard conditions
RHO_AIR_STD = 1.2232     # kg/m3 at 60 F, 14.696 psia
STB_TO_M3 = 0.158987
SCF_TO_M3 = 0.0283168
PSI_TO_PA = 6894.76

def oil_sg(api):
    return 141.5 / (131.5 + api)

def standing_rs(p_psia, t_f, api, gas_sg):
    """Solution gas-oil ratio (scf/STB), Standing (1947)."""
    return gas_sg * ((p_psia / 18.2 + 1.4) * 10 ** (0.0125 * api - 0.00091 * t_f)) ** 1.2048

def standing_bo(rs, t_f, api, gas_sg):
    """Oil formation volume factor (bbl/STB), Standing (1947)."""
    return 0.9759 + 0.00012 * (rs * np.sqrt(gas_sg / oil_sg(api)) + 1.25 * t_f) ** 1.2

def gas_z(p_psia, t_f, gas_sg):
    """Gas compressibility factor: Sutton pseudo-criticals with the Papay correlation."""
    tpc = 169.2 + 349.5 * gas_sg - 74.0 * gas_sg ** 2
    ppc = 756.8 - 131.0 * gas_sg - 3.6 * gas_sg ** 2
    tpr = (t_f + 459.67) / tpc
    ppr = p_psia / ppc
    return 1 - 3.52 * ppr / 10 ** (0.9813 * tpr) + 0.274 * ppr ** 2 / 10 ** (0.8157 * tpr)

def gas_density(p_psia, t_f, gas_sg):
    """In-situ gas density (kg/m3) from the real-gas law."""
    z = gas_z(p_psia, t_f, gas_sg)
    molar_mass = 28.97 * gas_sg                  # kg/kmol
    t_k = (t_f - 32.0) * 5.0 / 9.0 + 273.15
    return p_psia * PSI_TO_PA * molar_mass / (z * 8314.46 * t_k)
