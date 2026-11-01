import numpy as np
import pandas as pd
from src.data.well_tests import add_well_test_features, calibrate_to_last_test
from src.models import perkins
from src.models.empirical import fit_choke_curve, predict_empirical, choke_pct_to_64ths

def _toy_wells():
    rows = []
    for well, start in [('A', '2020-01-01'), ('B', '2020-03-01')]:
        dates = pd.date_range(start, periods=90, freq='D')
        segment = (np.arange(90) >= 50).astype(int) if well == 'A' else np.zeros(90, int)   # A restarts on day 50
        rows.append(pd.DataFrame({'Well': well, 'Date': dates, 'Segment': segment,
                                  'Q_liquid_STB': 1000.0 + np.arange(90), 'WHP_psig': 500.0, 'Pdown_psig': 300.0,
                                  'Choke_pct': 40.0, 'Water_Cut': 0.3}))
    df = pd.concat(rows, ignore_index=True)
    df['Days_since_event'] = (df.Date - df.groupby(['Well', 'Segment']).Date.transform('min')).dt.days
    return df

def test_well_test_is_always_an_earlier_day_in_the_same_segment():
    df = add_well_test_features(_toy_wells())
    has = df.Test_date.notna()
    assert (df.loc[has, 'Test_date'] < df.loc[has, 'Date']).all()           # never the same day: no leakage
    test_rows = df.set_index(['Well', 'Date'])
    for row in df[has].itertuples():
        assert test_rows.loc[(row.Well, row.Test_date), 'Segment'] == row.Segment
        assert test_rows.loc[(row.Well, row.Test_date), 'Is_test_day']

def test_tests_restart_with_each_segment_and_repeat_every_30_days():
    df = add_well_test_features(_toy_wells())
    a = df[df.Well == 'A']
    assert list(a.loc[a.Is_test_day, 'Days_since_event']) == [0, 30, 0, 30]
    assert a.loc[a.Days_since_event == 0, 'Test_date'].isna().all()          # first day of a segment has no prior test

def test_calibration_removes_a_constant_bias():
    df = add_well_test_features(_toy_wells())
    biased = pd.Series(df.Q_liquid_STB.values * 1.3, index=df.index)
    calibrated = calibrate_to_last_test(df, biased)
    has = df.Test_date.notna()
    # Rate drifts by ~1 STB/d per day, so after removing the 30% bias the remaining error is tiny
    assert np.max(np.abs(calibrated[has] / df.Q_liquid_STB[has] - 1)) < 0.05

def test_choke_curve_fits_recover_known_exponents():
    rng = np.random.default_rng(0)
    pct = rng.uniform(5, 95, 300)
    q_full = rng.uniform(5000, 20000, 300)
    factor, n = perkins.fit_choke_curve(perkins.apply_choke_curve(q_full, pct, 0.7, 1.4), q_full, pct)
    assert abs(factor - 0.7) < 1e-6 and abs(n - 1.4) < 1e-6

    p, glr = rng.uniform(300, 1500, 300), rng.uniform(100, 1500, 300)
    q = predict_empirical('Gilbert', p, glr, choke_pct_to_64ths(pct, 72.0, 0.8))
    s_max, n = fit_choke_curve('Gilbert', p, glr, pct, q)
    assert abs(s_max - 72.0) < 1e-6 and abs(n - 0.8) < 1e-6
