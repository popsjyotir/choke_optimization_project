import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold
from src.data.well_tests import add_well_test_features, calibrate_to_last_test, TEST_FEATURES
from src.models.empirical import CORRELATIONS, predict_empirical, choke_pct_to_64ths, fit_choke_curve
from src.models import perkins
from src.models.ml import (VOLVE_FEATURES, DABIRI_FEATURES, MODELS, fit_ml, fit_hybrid, predict_hybrid,
                           fit_quantiles)

TRAIN_FRACTION = 0.7  # chronological split: first 70% of each well's history trains
N_TIME_BLOCKS = 5     # tuning folds for the chronological split hold out whole time blocks
SETTLE_DAYS = 14      # days after a restart or workover before a well counts as settled
AREA_EXPONENT = 1.0   # flow area ∝ (% open)^n; fitted n is only reported, as it varies 0.65-3.8 between wells
WT_FEATURES = VOLVE_FEATURES + TEST_FEATURES

def metrics(q_obs, q_pred):
    """Relative errors in %, defined as (predicted - observed) / observed, plus RMSE and R2."""
    q_obs, q_pred = np.asarray(q_obs), np.asarray(q_pred)
    rel = (q_pred - q_obs) / q_obs * 100
    return {
        'n': len(q_obs),
        'AAPRE_%': np.mean(np.abs(rel)),
        'MedAPE_%': np.median(np.abs(rel)),
        'P90APE_%': np.percentile(np.abs(rel), 90),
        'APRE_%': np.mean(rel),
        'RMSE': np.sqrt(np.mean((q_pred - q_obs) ** 2)),
        'R2': 1 - np.sum((q_obs - q_pred) ** 2) / np.sum((q_obs - q_obs.mean()) ** 2),
    }

SUBSETS = {
    'All': lambda g: g,
    'Choke-controlled': lambda g: g[g.Choke_controlled],
    'Not choke-controlled': lambda g: g[~g.Choke_controlled],
    'Critical': lambda g: g[g.Regime == 'Critical'],
    'Subcritical': lambda g: g[g.Regime == 'Subcritical'],
    'Settled': lambda g: g[g.Days_since_event >= SETTLE_DAYS],
}

def volve_splits(df):
    """Yields (split, fold, train_idx, test_idx) for leave-one-well-out and chronological splits."""
    for well in df.Well.unique():
        yield 'Leave-one-well-out', well, df.index[df.Well != well], df.index[df.Well == well]
    rank = df.groupby('Well').Date.rank(method='first', pct=True)
    yield 'Chronological', 'all wells', df.index[rank <= TRAIN_FRACTION], df.index[rank > TRAIN_FRACTION]

def tuning_groups(split, train):
    """Wells for leave-one-well-out; consecutive time blocks within each well for the chronological split."""
    if split == 'Leave-one-well-out':
        return train.Well.values
    rank = train.groupby('Well').Date.rank(method='first', pct=True)
    return np.minimum((rank * N_TIME_BLOCKS).astype(int), N_TIME_BLOCKS - 1).values

def evaluate_volve(data_path="data/volve_choke.csv"):
    df = add_well_test_features(pd.read_csv(data_path, parse_dates=['Date'])).reset_index(drop=True)
    y = df.Q_liquid_STB.values

    # Perkins at a fully-open 1-inch reference bore with Cd = 0.826; a fitted flow factor scales it per fold
    q_full, r_c, is_critical = perkins.predict_perkins(df.WHP_psig, df.Pdown_psig, df.WHT_F, df.GOR_scf_stb,
                                                       df.Water_Cut, 64.0)
    df['Regime'] = np.where(is_critical, 'Critical', 'Subcritical')

    # Diagnostic: the area exponent each well's own data implies (choke hardware differs between wells)
    curves = [{'Split': 'Per well (all data)', 'Fold': w, 'Model': 'Perkins',
               'Fitted_exponent_n': perkins.fit_choke_curve(y[g], q_full[g], df.Choke_pct[g])[1]}
              for w, g in df.groupby('Well').indices.items()]
    records, extras = [], {}
    for split, fold, tr, te in volve_splits(df):
        print(f"  {split}: test {fold} ({len(te)} rows)")
        train, test = df.loc[tr], df.loc[te]
        groups = tuning_groups(split, train)
        test_wells = df.index[df.Well.isin(test.Well.unique())]   # for test-day calibration
        preds, full = {}, {}

        for name in CORRELATIONS:
            args = (name, train.WHP_psig, train.GLR_scf_stb, train.Choke_pct, train.Q_liquid_STB)
            s_max, _ = fit_choke_curve(*args, n=AREA_EXPONENT)
            preds[name] = predict_empirical(name, test.WHP_psig, test.GLR_scf_stb,
                                            choke_pct_to_64ths(test.Choke_pct, s_max, AREA_EXPONENT))
            curves.append({'Split': split, 'Fold': fold, 'Model': name, 'S_max_64ths': s_max,
                           'Fitted_exponent_n': fit_choke_curve(*args)[1]})

        factor = perkins.fit_flow_factor(y[tr], perkins.apply_choke_curve(q_full[tr], train.Choke_pct, 1.0, AREA_EXPONENT))
        curves.append({'Split': split, 'Fold': fold, 'Model': 'Perkins', 'Flow_factor': factor,
                       'Fitted_exponent_n': perkins.fit_choke_curve(y[tr], q_full[tr], train.Choke_pct)[1]})
        q_perk = perkins.apply_choke_curve(q_full, df.Choke_pct, factor, AREA_EXPONENT)
        preds['Perkins'] = q_perk[te]
        full['Perkins'] = q_perk[test_wells]
        if split == 'Chronological':
            # Engineer's practice: calibrate Cd on each well's own history
            per_well = pd.Series(q_perk[te], index=te)
            for well in test.Well.unique():
                w_tr, w_te = tr[train.Well == well], te[test.Well == well]
                per_well[w_te] *= perkins.fit_flow_factor(y[w_tr], q_perk[w_tr])
            preds['Perkins (per-well Cd)'] = per_well.values

        for name in MODELS:
            model = fit_ml(name, train[VOLVE_FEATURES], y[tr], groups)
            preds[name] = model.predict(test[VOLVE_FEATURES])
            if name == 'Gradient Boosting':
                full[name] = model.predict(df.loc[test_wells, VOLVE_FEATURES])
        hybrid = fit_hybrid(train[VOLVE_FEATURES], y[tr], q_perk[tr], groups)
        preds['Hybrid (Perkins + GB)'] = predict_hybrid(hybrid, test[VOLVE_FEATURES], q_perk[te])
        full['Hybrid (Perkins + GB)'] = predict_hybrid(hybrid, df.loc[test_wells, VOLVE_FEATURES], q_perk[test_wells])

        # Virtual-flow-meter variants that use the last periodic well test
        gb_wt = fit_ml('Gradient Boosting', train[WT_FEATURES], y[tr], groups)
        preds['GB + well tests'] = gb_wt.predict(test[WT_FEATURES])
        hyb_wt = fit_hybrid(train[WT_FEATURES], y[tr], q_perk[tr], groups)
        preds['Hybrid + well tests'] = predict_hybrid(hyb_wt, test[WT_FEATURES], q_perk[te])
        for name, q in full.items():
            calibrated = calibrate_to_last_test(df.loc[test_wells], pd.Series(q, index=test_wells))
            preds[f'{name} + test calibration'] = calibrated[te].values

        if split == 'Chronological':
            extras = _chronological_extras(train, test, y[tr], gb_wt)

        for model_name, q_pred in preds.items():
            records.append(pd.DataFrame({'Split': split, 'Fold': fold, 'Model': model_name, 'Row': te,
                                         'Well': test.Well.values, 'Regime': test.Regime.values,
                                         'Choke_controlled': test.Choke_controlled.values,
                                         'Days_since_event': test.Days_since_event.values,
                                         'Water_Cut': test.Water_Cut.values,
                                         'Q_obs': y[te], 'Q_pred': np.asarray(q_pred)}))

    pred = pd.concat(records, ignore_index=True)
    pred.to_csv("outputs/volve_predictions.csv", index=False)
    pd.DataFrame(curves).to_csv("outputs/choke_curve_fits.csv", index=False)

    rows = []
    for (split, model), g in pred.groupby(['Split', 'Model'], sort=False):
        for subset, select in SUBSETS.items():
            s = select(g)
            if len(s):
                rows.append({'Split': split, 'Model': model, 'Subset': subset, **metrics(s.Q_obs, s.Q_pred)})
    table = pd.DataFrame(rows)
    table.to_csv("outputs/volve_metrics.csv", index=False)

    per_well = pd.DataFrame([{'Split': s, 'Model': m, 'Well': w, **metrics(g.Q_obs, g.Q_pred)}
                             for (s, m, w), g in pred.groupby(['Split', 'Model', 'Well'], sort=False)])
    per_well.to_csv("outputs/volve_metrics_per_well.csv", index=False)
    return table, extras

def conformal_intervals(train, y_train, test, calib_fraction=0.2, coverage=0.8):
    """Conformalized quantile regression (CQR) in log space, calibrated on the latest part of the training data.

    Quantile models are fitted on the earlier training rows; on the most recent `calib_fraction` of each well's
    training history, the conformity score max(log lo - log y, log y - log hi) gives the widening needed for the
    target coverage. Using the latest rows lets the correction see the drift that the test period will have.
    """
    rank = train.groupby('Well').Date.rank(method='first', pct=True).values
    proper, calib = rank <= 1 - calib_fraction, rank > 1 - calib_fraction
    alpha = (1 - coverage) / 2
    quantiles = fit_quantiles(train.loc[proper, WT_FEATURES], y_train[proper], quantiles=(alpha, 1 - alpha))
    lo_c, hi_c = (np.log(quantiles[q].predict(train.loc[calib, WT_FEATURES])) for q in (alpha, 1 - alpha))
    scores = np.maximum(lo_c - np.log(y_train[calib]), np.log(y_train[calib]) - hi_c)
    n = calib.sum()
    widen = np.quantile(scores, min(1.0, np.ceil((n + 1) * coverage) / n))
    lo, hi = (quantiles[q].predict(test[WT_FEATURES]) for q in (alpha, 1 - alpha))
    return lo, hi, lo * np.exp(-widen), hi * np.exp(widen)

def _chronological_extras(train, test, y_train, gb_wt):
    """Prediction intervals and permutation feature importance for the well-test model on the chronological split."""
    raw_lo, raw_hi, lo, hi = conformal_intervals(train, y_train, test)
    q_obs = test.Q_liquid_STB.values
    intervals = pd.DataFrame({'Well': test.Well.values, 'Date': test.Date.values, 'Q_obs': q_obs,
                              'Q_pred': gb_wt.predict(test[WT_FEATURES]), 'Q_p10': lo, 'Q_p90': hi})
    intervals.to_csv("outputs/prediction_intervals.csv", index=False)
    coverage = {'Nominal_%': 80.0,
                'Raw_quantile_coverage_%': np.mean((q_obs >= raw_lo) & (q_obs <= raw_hi)) * 100,
                'Conformal_coverage_%': np.mean((q_obs >= lo) & (q_obs <= hi)) * 100,
                'Median_width_%_of_pred': np.median((hi - lo) / intervals.Q_pred) * 100}
    pd.DataFrame([coverage]).to_csv("outputs/prediction_interval_coverage.csv", index=False)

    imp = permutation_importance(gb_wt, test[WT_FEATURES], q_obs, scoring='neg_mean_absolute_percentage_error',
                                 n_repeats=5, random_state=42, n_jobs=-1)
    importance = pd.DataFrame({'Feature': WT_FEATURES, 'Importance': imp.importances_mean * 100,
                               'Std': imp.importances_std * 100}).sort_values('Importance', ascending=False)
    importance.to_csv("outputs/feature_importance.csv", index=False)
    return {'coverage': coverage, 'importance': importance}

def evaluate_dabiri(data_path="data/dabiri_2024.csv", n_repeats=3):
    """Empirical correlations as published, and ML with repeated 5-fold CV, with and without duplicate rows."""
    raw = pd.read_csv(data_path)
    rows = []
    for name in CORRELATIONS:
        q = predict_empirical(name, raw.Pwh_psi, raw.GLR_scf_stb, raw.Choke_64ths)
        rows.append({'Model': name, 'Data': 'All 565 rows', **metrics(raw.Q_liquid_STB, q)})

    for label, df in [('All 565 rows', raw), ('Duplicates removed', raw.drop_duplicates(subset=DABIRI_FEATURES + ['Q_liquid_STB']))]:
        df = df.reset_index(drop=True)
        for name in MODELS:
            scores = []
            for seed in range(n_repeats):
                oof = np.zeros(len(df))
                for tr, te in KFold(5, shuffle=True, random_state=seed).split(df):
                    model = fit_ml(name, df.loc[tr, DABIRI_FEATURES], df.Q_liquid_STB.values[tr])
                    oof[te] = model.predict(df.loc[te, DABIRI_FEATURES])
                scores.append(metrics(df.Q_liquid_STB, oof))
            rows.append({'Model': name, 'Data': label, **pd.DataFrame(scores).mean().to_dict()})

    table = pd.DataFrame(rows)
    table.to_csv("outputs/dabiri_metrics.csv", index=False)
    return table

if __name__ == "__main__":
    pd.set_option('display.width', 200)
    print("Dabiri et al. (2024) benchmark")
    print(evaluate_dabiri().round(2).to_string(index=False))
    print("\nVolve benchmark")
    table, extras = evaluate_volve()
    for metric in ['AAPRE_%', 'MedAPE_%']:
        print(f"\n{metric}")
        print(table[table.Subset.isin(['All', 'Choke-controlled', 'Settled'])]
              .pivot_table(index='Model', columns=['Split', 'Subset'], values=metric, sort=False).round(1).to_string())
    print("\nPrediction interval (GB + well tests, chronological):", {k: round(v, 1) for k, v in extras['coverage'].items()})
    print("\nPermutation importance (increase in MAPE, %):")
    print(extras['importance'].round(2).to_string(index=False))
