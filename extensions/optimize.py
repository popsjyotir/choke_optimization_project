import numpy as np
import pandas as pd
import joblib
from src.models.ml import VOLVE_FEATURES, fit_ml, fit_hybrid
from src.models.perkins import fit_flow_factor
from extensions.predictors import perkins_reference, make_predictors
from extensions.system.vlp import DataVLP
from extensions.system.ipr import LinearIPR, fit_pi
from extensions.system.nodal import choke_rate_function, solve_operating_point

CHOKE_GRID = np.arange(5.0, 99.0, 1.0)  # % open
STATE_COLS = ['WHP_psig', 'Pdown_psig', 'WHT_F', 'GOR_scf_stb', 'GLR_scf_stb', 'Water_Cut']

def train_final_model(data_path="data/volve_choke.csv", model_path="models/extensions/whatif_model.pkl"):
    """Fits choke models on all Volve data, plus a VLP and productivity index for each well with downhole data."""
    df = pd.read_csv(data_path, parse_dates=['Date'])
    y, groups = df.Q_liquid_STB.values, df.Well.values
    q_ref, _, _ = perkins_reference(df)
    factor = fit_flow_factor(y, q_ref)

    wells = {}
    for well, g in df.groupby('Well'):
        vlp, pi = DataVLP.fit(g), fit_pi(g)
        if vlp is not None and pi is not None:
            wells[well] = {'vlp': vlp, 'pi': pi}

    bundle = {
        'perkins_factor': factor,
        'hybrid': fit_hybrid(df[VOLVE_FEATURES], y, q_ref * factor, groups),
        'gradient_boosting': fit_ml('Gradient Boosting', df[VOLVE_FEATURES], y, groups),
        'wells': wells,
        'coverage': df[['Well', 'Choke_pct', 'WHP_psig']],
    }
    joblib.dump(bundle, model_path)
    print(f"Trained choke models on {len(df)} rows; nodal models for wells: {', '.join(wells) or 'none'}.")
    print(f"Saved to '{model_path}'.")
    return bundle

def rate_vs_choke(state, choke_now, q_now, pwf_now=None, well=None, vlp=None, ipr=None,
                  model="Hybrid (Perkins + GB)", model_path="models/extensions/whatif_model.pkl"):
    """Liquid rate (and WHP, Pwf) across choke openings, matched to today's measured rate.

    Nodal mode is used when a VLP and IPR are available: passed in, or fitted for `well` (needs pwf_now).
    Otherwise WHP is held at today's value (fixed-WHP mode) and a warning is printed.
    Returns (curve DataFrame, mode).
    """
    bundle = joblib.load(model_path)
    predict = make_predictors(bundle['perkins_factor'], bundle['hybrid'], bundle['gradient_boosting'])[model]
    fitted = bundle['wells'].get(well, {})
    vlp = vlp or fitted.get('vlp')
    if ipr is None and 'pi' in fitted and pwf_now is not None:
        ipr = LinearIPR(fitted['pi'], q_now, pwf_now)

    if vlp is not None and ipr is not None:
        mode, rows = 'Nodal', []
        for c in CHOKE_GRID:
            sol = solve_operating_point(choke_rate_function(predict, state, c), vlp, ipr, state['Pdown_psig'], state)
            rows.append({'Choke_pct': c, **({'Q_liquid_STB': sol['q'], 'WHP_psig': sol['whp'], 'Pwf_psig': sol['pwf']}
                                           if sol else {'Q_liquid_STB': np.nan, 'WHP_psig': np.nan, 'Pwf_psig': np.nan})})
        curve = pd.DataFrame(rows)
        now = solve_operating_point(choke_rate_function(predict, state, choke_now), vlp, ipr, state['Pdown_psig'], state)
        q_model_now = now['q'] if now else np.nan
    else:
        mode = 'Fixed WHP'
        print("-> Warning: no VLP/IPR for this well; holding WHP fixed as the choke moves (approximate).")
        curve = pd.DataFrame({'Choke_pct': CHOKE_GRID, 'WHP_psig': state['WHP_psig'], 'Pwf_psig': np.nan})
        curve['Q_liquid_STB'] = [choke_rate_function(predict, state, c)(state['WHP_psig'])[0] for c in CHOKE_GRID]
        q_model_now = choke_rate_function(predict, state, choke_now)(state['WHP_psig'])[0]

    # Match the model to today's measurement so its level bias does not shift the recommendation
    match = q_now / q_model_now if q_model_now and np.isfinite(q_model_now) else 1.0
    curve['Q_liquid_STB'] *= match
    return curve, mode

def find_optimal_choke(target_q_liquid, state, choke_now, q_now, pwf_now=None, well=None, vlp=None, ipr=None,
                       model_path="models/extensions/whatif_model.pkl"):
    """Recommends a choke opening (%) for a target liquid rate.

    state: today's WHP_psig, Pdown_psig, WHT_F, GOR_scf_stb, GLR_scf_stb, Water_Cut.
    Pass your own `vlp` (e.g. TableVLP.from_csv(...)) and/or `ipr` to override the fitted ones.
    """
    curve, mode = rate_vs_choke(state, choke_now, q_now, pwf_now, well, vlp, ipr, model_path=model_path)
    valid = curve.dropna(subset=['Q_liquid_STB'])
    print(f"-> Mode: {mode}. Matched to today's rate of {q_now:.0f} STB/d at {choke_now:.1f}% open.")
    if valid.empty:
        print("-> The well cannot flow at any choke opening under these conditions.")
        return None, curve
    print(f"-> Reachable range: {valid.Q_liquid_STB.min():.0f} - {valid.Q_liquid_STB.max():.0f} STB/d")
    if not valid.Q_liquid_STB.min() <= target_q_liquid <= valid.Q_liquid_STB.max():
        print(f"-> Target {target_q_liquid:.0f} STB/d is outside the reachable range.")

    best = valid.iloc[int(np.argmin(np.abs(valid.Q_liquid_STB - target_q_liquid)))]
    print(f"-> Recommended choke: {best.Choke_pct:.0f}% open -> {best.Q_liquid_STB:.0f} STB/d, WHP {best.WHP_psig:.0f} psig"
          + (f", Pwf {best.Pwf_psig:.0f} psig" if np.isfinite(best.Pwf_psig) else ""))

    # Data coverage: has this well operated near the recommended choke and WHP before?
    cov = joblib.load(model_path)['coverage']
    cov = cov[cov.Well == well] if well in set(cov.Well) else cov
    near = ((cov.Choke_pct - best.Choke_pct).abs() <= 5) & ((cov.WHP_psig - best.WHP_psig).abs() <= 0.15 * best.WHP_psig)
    n_near = int(near.sum())
    print(f"-> Data coverage: {n_near} past days within 5 points of choke and 15% of WHP"
          + (" (LOW CONFIDENCE: few comparable days)" if n_near < 10 else ""))
    return best.Choke_pct, curve

if __name__ == "__main__":
    train_final_model()

    # Example: the latest steady day of well F-14 H, asking for 15% more liquid
    df = pd.read_csv("data/volve_choke.csv", parse_dates=['Date'])
    today = df[(df.Well == 'F-14 H') & df.Steady & df.Pwf_psig.notna()].iloc[-1]
    state = {c: today[c] for c in STATE_COLS}
    print(f"\nF-14 H on {today.Date.date()}: {today.Q_liquid_STB:.0f} STB/d at {today.Choke_pct:.1f}% open, "
          f"WHP {today.WHP_psig:.0f} psig, Pwf {today.Pwf_psig:.0f} psig")
    for label, kwargs in [("Nodal", {'pwf_now': today.Pwf_psig}), ("Fixed WHP (no downhole data)", {})]:
        print(f"\n{label}:")
        _, curve = find_optimal_choke(1.15 * today.Q_liquid_STB, state, today.Choke_pct, today.Q_liquid_STB,
                                      well='F-14 H', **kwargs)
        curve.to_csv(f"outputs/extensions/rate_vs_choke_F-14H_{'nodal' if kwargs else 'fixed'}.csv", index=False)
