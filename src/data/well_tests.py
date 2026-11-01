import numpy as np
import pandas as pd

TEST_INTERVAL_DAYS = 30
TEST_FEATURES = ['log_q_test', 'Days_since_test', 'dWHP_since_test', 'dPdown_since_test',
                 'dChoke_since_test', 'dWC_since_test']

def add_well_test_features(df, interval_days=TEST_INTERVAL_DAYS):
    """Simulates the periodic well tests a real virtual flow meter is calibrated with.

    A test is assumed on the first day of each well segment (after a restart or workover) and then every
    `interval_days`. Each row gets the most recent test strictly before it: the measured rate and how
    WHP, downstream pressure, choke and water cut have changed since. Rows with no earlier test get NaN.
    The test-day rate is a past measurement; the current day's rate is never used.
    """
    df = df.sort_values(['Well', 'Date']).copy()
    df['Is_test_day'] = df.groupby(['Well', 'Segment', df.Days_since_event // interval_days]).Date.transform('min') == df.Date
    tests = df.loc[df.Is_test_day, ['Well', 'Segment', 'Date', 'Q_liquid_STB', 'WHP_psig', 'Pdown_psig', 'Choke_pct', 'Water_Cut']]
    tests = tests.rename(columns={c: f'test_{c}' for c in tests.columns if c not in ('Well', 'Segment')})

    merged = pd.merge_asof(df.sort_values('Date'), tests.sort_values('test_Date'), left_on='Date', right_on='test_Date',
                           by=['Well', 'Segment'], allow_exact_matches=False)
    merged = merged.sort_values(['Well', 'Date']).set_index(df.index)
    df['Test_date'] = merged.test_Date
    df['Q_test_STB'] = merged.test_Q_liquid_STB
    df['log_q_test'] = np.log(merged.test_Q_liquid_STB)
    df['Days_since_test'] = (df.Date - merged.test_Date).dt.days
    df['dWHP_since_test'] = df.WHP_psig - merged.test_WHP_psig
    df['dPdown_since_test'] = df.Pdown_psig - merged.test_Pdown_psig
    df['dChoke_since_test'] = df.Choke_pct - merged.test_Choke_pct
    df['dWC_since_test'] = df.Water_Cut - merged.test_Water_Cut
    return df

def calibrate_to_last_test(df, q_pred, clip=(0.5, 2.0)):
    """Scales each prediction by measured / predicted rate on the row's last test day (per-well bias correction).

    q_pred must be a Series aligned with df covering the test days used. Rows without an earlier test keep
    the raw prediction.
    """
    on_test = df.Is_test_day & q_pred.notna()
    ratio = (df.Q_liquid_STB / q_pred)[on_test].clip(*clip)
    key = pd.Series(ratio.values, index=pd.MultiIndex.from_frame(df.loc[on_test, ['Well', 'Date']]))
    lookup = pd.MultiIndex.from_frame(df[['Well', 'Test_date']].rename(columns={'Test_date': 'Date'}))
    factor = key.reindex(lookup).fillna(1.0).values
    return q_pred * factor
