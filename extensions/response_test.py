import numpy as np
import pandas as pd
from src.models.ml import VOLVE_FEATURES, fit_ml, fit_hybrid
from src.models.perkins import fit_flow_factor
from extensions.predictors import perkins_reference, make_predictors
from extensions.system.vlp import DataVLP
from extensions.system.ipr import anchored_ipr
from extensions.system.nodal import choke_rate_function, solve_operating_point

MIN_CHOKE_MOVE = 2.0  # % points
N_BLOCKS = 5          # blocked time cross-validation folds
PURGE_DAYS = 14       # training rows this close to a test block are dropped
STATE_COLS = ['WHP_psig', 'Pdown_psig', 'WHT_F', 'GOR_scf_stb', 'GLR_scf_stb', 'Water_Cut']

def time_blocks(df, n_blocks=N_BLOCKS):
    """Assigns each row to one of n consecutive time blocks within its well."""
    rank = df.groupby('Well').Date.rank(method='first', pct=True)
    return np.minimum((rank * n_blocks).astype(int), n_blocks - 1)

def purged_train_mask(df, block, k, purge_days=PURGE_DAYS):
    """Rows outside block k and more than purge_days from block k's dates in the same well."""
    test = block == k
    span = df[test].groupby('Well').Date.agg(['min', 'max'])
    lo = df.Well.map(span['min']) - pd.Timedelta(days=purge_days)
    hi = df.Well.map(span['max']) + pd.Timedelta(days=purge_days)
    near = lo.notna() & (df.Date >= lo) & (df.Date <= hi)
    return ~test & ~near

def fit_predictors(train):
    """Fits the Perkins factor, hybrid and Gradient Boosting choke models on a training set."""
    q_ref, _, _ = perkins_reference(train)
    y, groups = train.Q_liquid_STB.values, train.Well.values
    factor = fit_flow_factor(y, q_ref)
    return make_predictors(factor,
                           hybrid=fit_hybrid(train[VOLVE_FEATURES], y, q_ref * factor, groups),
                           gradient_boosting=fit_ml('Gradient Boosting', train[VOLVE_FEATURES], y, groups))

def wilson_interval(successes, n, z=1.96):
    """95% Wilson score interval for a proportion, in %."""
    if n == 0:
        return np.nan, np.nan
    p = successes / n
    centre = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return (centre - half) * 100, (centre + half) * 100

def find_choke_events(df, min_move=MIN_CHOKE_MOVE):
    """Pairs a steady day before a choke move with the first steady day after it, in the same well.

    Day 1 (before) and day 3 (after) bracket the move on day 2, whose daily averages mix both states.
    Both days need a downhole pressure reading.
    """
    events = []
    for well, g in df.dropna(subset=['Pwf_psig']).groupby('Well'):
        g = g.set_index('Date').sort_index()
        for day2, row in g[g.Choke_change_pct.abs() >= min_move].iterrows():
            day1, day3 = day2 - pd.Timedelta(days=1), day2 + pd.Timedelta(days=1)
            if day1 in g.index and day3 in g.index and g.loc[day1, 'Steady'] and g.loc[day3, 'Steady']:
                events.append({'Well': well, 'before': day1, 'after': day3})
    return pd.DataFrame(events)

def choke_response_test(data_path="data/volve_choke.csv"):
    """Do models predict how rate (and WHP) change after a real choke move?

    Choke models are evaluated with blocked time cross-validation: each well's history is cut into
    N_BLOCKS consecutive blocks, and events in a block are predicted by models trained on the other
    blocks with a PURGE_DAYS gap. Fixed-WHP mode moves only the choke. Nodal mode couples the choke model
    with an anchored IPR and a data-fitted VLP, both fitted on that well's data before the event, so WHP is
    solved, not held. Predicted changes are differences between model runs, so level bias cancels out.
    """
    df = pd.read_csv(data_path, parse_dates=['Date'])
    block = time_blocks(df)
    events = find_choke_events(df)
    lookup = df.set_index(['Well', 'Date'])
    events['Block'] = [block[(df.Well == e.Well) & (df.Date == e.before)].iloc[0] for e in events.itertuples()]
    print(f"  {len(events)} choke-move events across {events.Well.nunique()} wells")

    rows = []
    for k in range(N_BLOCKS):
        fold_events = events[events.Block == k]
        if fold_events.empty:
            continue
        print(f"  Block {k + 1}/{N_BLOCKS}: {len(fold_events)} events")
        predictors = fit_predictors(df[purged_train_mask(df, block, k)])
        rows += _evaluate_events(fold_events, df, lookup, predictors)

    detail = pd.DataFrame(rows)
    detail.to_csv("outputs/extensions/choke_response_events.csv", index=False)
    summary = _summarize(detail)
    summary.to_csv("outputs/extensions/choke_response_summary.csv", index=False)
    return summary

def _evaluate_events(events, df, lookup, predictors):
    rows = []
    for ev in events.itertuples():
        b, a = lookup.loc[(ev.Well, ev.before)], lookup.loc[(ev.Well, ev.after)]
        history = df[(df.Well == ev.Well) & (df.Date <= ev.before)]
        vlp = DataVLP.fit(history)
        ipr = anchored_ipr(history, b.Q_liquid_STB, b.Pwf_psig)
        state = {c: b[c] for c in STATE_COLS}
        record = {'Well': ev.Well, 'Date': ev.after, 'Choke_before': b.Choke_pct, 'Choke_after': a.Choke_pct,
                  'Q_before': b.Q_liquid_STB, 'dQ_obs': a.Q_liquid_STB - b.Q_liquid_STB,
                  'WHP_before': b.WHP_psig, 'dWHP_obs': a.WHP_psig - b.WHP_psig,
                  'PI': getattr(ipr, 'pi', np.nan)}

        for name, predict in predictors.items():
            q_before = choke_rate_function(predict, state, b.Choke_pct)(b.WHP_psig)[0]
            q_after = choke_rate_function(predict, state, a.Choke_pct)(b.WHP_psig)[0]
            record[f'{name}|Fixed WHP|dQ'] = q_after - q_before

            if vlp is not None and ipr is not None:
                sol_b, sol_a = (solve_operating_point(choke_rate_function(predict, state, c), vlp, ipr, b.Pdown_psig, state)
                                for c in (b.Choke_pct, a.Choke_pct))
                if sol_b and sol_a:
                    record[f'{name}|Nodal|dQ'] = sol_a['q'] - sol_b['q']
                    record[f'{name}|Nodal|dWHP'] = sol_a['whp'] - sol_b['whp']
        rows.append(record)
    return rows

def _summarize(detail):
    summary = []
    for col in [c for c in detail.columns if c.endswith('|dQ')]:
        model, mode, _ = col.split('|')
        s = detail.dropna(subset=[col])
        hits = int(np.sum(np.sign(s[col]) == np.sign(s.dQ_obs)))
        lo, hi = wilson_interval(hits, len(s))
        rec = {'Model': model, 'Mode': mode, 'n': len(s),
               'Direction_correct_%': hits / len(s) * 100 if len(s) else np.nan,
               'Direction_CI_low': lo, 'Direction_CI_high': hi,
               'dQ_error_%_of_rate': np.mean(np.abs(s[col] - s.dQ_obs) / s.Q_before) * 100}
        wcol = f'{model}|{mode}|dWHP'
        if wcol in detail:
            rec['WHP_direction_correct_%'] = np.mean(np.sign(s[wcol]) == np.sign(s.dWHP_obs)) * 100
            rec['dWHP_MAE_psi'] = np.mean(np.abs(s[wcol] - s.dWHP_obs))
        summary.append(rec)
    no_change = {'Model': 'No-change baseline', 'Mode': '-', 'n': len(detail), 'Direction_correct_%': np.nan,
                 'dQ_error_%_of_rate': np.mean(np.abs(detail.dQ_obs) / detail.Q_before) * 100}
    hits = int(np.sum(np.sign(detail.Choke_after - detail.Choke_before) == np.sign(detail.dQ_obs)))
    lo, hi = wilson_interval(hits, len(detail))
    choke_sign = {'Model': 'Rate follows choke direction', 'Mode': '-', 'n': len(detail),
                  'Direction_correct_%': hits / len(detail) * 100, 'Direction_CI_low': lo, 'Direction_CI_high': hi}
    return pd.DataFrame(summary + [no_change, choke_sign])

if __name__ == "__main__":
    pd.set_option('display.width', 160)
    print("Choke-response test")
    print(choke_response_test().round(1).to_string(index=False))
