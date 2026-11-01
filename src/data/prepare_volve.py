import numpy as np
import pandas as pd

# Unit conversions
BAR_TO_PSI = 14.5038
SM3_TO_STB = 6.28981
SM3_TO_SCF = 35.3147
P_ATM_PSI = 14.696

CONTROLLED_RATIO = 0.90   # above this downstream/upstream ratio the choke drop is ~30 psi or less
SHUT_IN_DAYS = 7          # a restart after at least this many shut-in days starts a new well segment
WC_JUMP = 0.20            # a step of at least this size in 14-day median water cut starts a new segment

def _restart_days(df):
    """Flags the first producing day after a shut-in of at least SHUT_IN_DAYS days (raw daily rows)."""
    flags = pd.Series(False, index=df.index)
    for _, g in df.groupby('WELL_BORE_CODE'):
        down = g.ON_STREAM_HRS.fillna(0) < 1
        run = down.groupby((~down).cumsum()).cumsum()           # consecutive shut-in days so far
        flags.loc[g.index] = ~down & (run.shift().fillna(0) >= SHUT_IN_DAYS)
    return flags

def add_segments(out):
    """Splits each well's history at restarts and water-cut steps (e.g. workovers) into segments."""
    out = out.sort_values(['Well', 'Date']).copy()
    event = out.Restart.copy()
    for _, g in out.groupby('Well'):
        wc = g.set_index('Date').Water_Cut
        before = wc.rolling('14D').median()
        after = wc[::-1].rolling(14, min_periods=5).median()[::-1].shift(-1)
        jump = ((after - before).abs() >= WC_JUMP).values
        dates = g.Date.values
        last = None
        for i in np.flatnonzero(jump):                         # keep the first day of each cluster
            if last is None or (dates[i] - last) > np.timedelta64(30, 'D'):
                event.loc[g.index[min(i + 1, len(g) - 1)]] = True
                last = dates[i]
    out['Well_event'] = event
    out['Segment'] = out.groupby('Well').Well_event.cumsum().astype(int)
    out['Days_since_event'] = (out.Date - out.groupby(['Well', 'Segment']).Date.transform('min')).dt.days
    return out

def prepare_volve(xlsx_path="data/raw/Volve production data.xlsx", output_path="data/volve_choke.csv",
                  min_hours=20.0, max_choke_pct=99.0):
    """Cleans Volve daily production data into choke-performance records in field units.

    Assumptions:
    - Volve pressures (AVG_WHP_P, DP_CHOKE_SIZE) are gauge pressures in bar.
    - Downstream choke pressure = WHP - choke pressure drop.
    - Daily volumes are scaled to a 24-hour rate using ON_STREAM_HRS.
    - Days with the choke >= 99% open are dropped (choke not controlling flow).
    - Downhole pressure (Pwf) is kept only where the gauge reads above WHP; otherwise it is NaN.
    - The choke controls flow when the pressure ratio is <= CONTROLLED_RATIO.
    - Restarts after shut-ins and water-cut steps split each well into segments (Segment, Days_since_event).
    """
    df = pd.read_excel(xlsx_path, sheet_name="Daily Production Data")
    log = [("raw rows", len(df))]

    def keep(mask, label):
        nonlocal df
        df = df[mask].copy()
        log.append((label, len(df)))

    keep(df.WELL_TYPE == "OP", "producer wells only")

    # History-dependent fields are computed before day-level filtering
    df = df.sort_values(['WELL_BORE_CODE', 'DATEPRD'])
    df['Cum_liquid_STB'] = (df.BORE_OIL_VOL.fillna(0) + df.BORE_WAT_VOL.clip(lower=0).fillna(0)).groupby(df.WELL_BORE_CODE).cumsum() * SM3_TO_STB
    prev = df.groupby('WELL_BORE_CODE')[['DATEPRD', 'AVG_CHOKE_SIZE_P', 'ON_STREAM_HRS']].shift()
    prev_ok = ((df.DATEPRD - prev.DATEPRD).dt.days == 1) & (prev.ON_STREAM_HRS >= min_hours)
    df['Choke_change_pct'] = np.where(prev_ok, df.AVG_CHOKE_SIZE_P - prev.AVG_CHOKE_SIZE_P, np.nan)
    # Running restart count, so the first kept day after a restart still marks it even if the restart day is filtered
    df['Restart_count'] = _restart_days(df).groupby(df.WELL_BORE_CODE).cumsum()

    keep(df.ON_STREAM_HRS >= min_hours, f"on stream >= {min_hours:.0f} h")
    keep((df.AVG_WHP_P > 0) & df.DP_CHOKE_SIZE.notna() & (df.DP_CHOKE_SIZE > 0), "valid WHP and choke dP")
    keep((df.AVG_CHOKE_SIZE_P > 0) & (df.AVG_CHOKE_SIZE_P < max_choke_pct), f"choke 0-{max_choke_pct:.0f}% open")
    keep((df.BORE_OIL_VOL > 0) & (df.BORE_GAS_VOL > 0) & (df.BORE_WAT_VOL >= 0), "positive oil/gas, non-negative water")

    hours = df.ON_STREAM_HRS.clip(upper=24.0)
    scale = 24.0 / hours

    # Downhole gauge: Volve stores 0 when the gauge is down (and then DP_TUBING = WHP), so only Pwf > WHP is kept.
    # DP_TUBING equals Pwf - WHP in this dataset, so it is derived rather than read.
    pwf_bar = df.AVG_DOWNHOLE_PRESSURE.where(df.AVG_DOWNHOLE_PRESSURE > df.AVG_WHP_P)

    out = pd.DataFrame({
        'Well': df.WELL_BORE_CODE.str.replace("NO 15/9-", "", regex=False).values,
        'Date': df.DATEPRD.values,
        'WHP_psig': (df.AVG_WHP_P * BAR_TO_PSI).values,
        'Pdown_psig': ((df.AVG_WHP_P - df.DP_CHOKE_SIZE) * BAR_TO_PSI).values,
        'Choke_pct': df.AVG_CHOKE_SIZE_P.values,
        'WHT_F': (df.AVG_WHT_P * 9 / 5 + 32).values,
        'Q_oil_STB': (df.BORE_OIL_VOL * scale * SM3_TO_STB).values,
        'Q_water_STB': (df.BORE_WAT_VOL * scale * SM3_TO_STB).values,
        'Q_gas_scf': (df.BORE_GAS_VOL * scale * SM3_TO_SCF).values,
        'Pwf_psig': (pwf_bar * BAR_TO_PSI).values,
        'BHT_F': (df.AVG_DOWNHOLE_TEMPERATURE.where(pwf_bar.notna()) * 9 / 5 + 32).values,
        'Cum_liquid_STB': df.Cum_liquid_STB.values,
        'Choke_change_pct': df.Choke_change_pct.values,
        'Restart_count': df.Restart_count.values,
    })
    out['DP_tubing_psi'] = out.Pwf_psig - out.WHP_psig
    # Steady day: the previous day was on stream and the choke moved by less than 2 points
    out['Steady'] = out.Choke_change_pct.abs() < 2.0
    out['Q_liquid_STB'] = out.Q_oil_STB + out.Q_water_STB
    out['GOR_scf_stb'] = out.Q_gas_scf / out.Q_oil_STB
    out['GLR_scf_stb'] = out.Q_gas_scf / out.Q_liquid_STB
    out['Water_Cut'] = out.Q_water_STB / out.Q_liquid_STB
    out['Pressure_Ratio'] = (out.Pdown_psig + P_ATM_PSI) / (out.WHP_psig + P_ATM_PSI)

    df = out
    keep((out.Pdown_psig > 0) & (out.Pressure_Ratio > 0) & (out.Pressure_Ratio < 1) & out.WHT_F.notna(),
         "physical downstream pressure and ratio")
    out = df.sort_values(['Well', 'Date']).reset_index(drop=True)

    # Choke controls the flow only when there is a real pressure drop across it
    out['Choke_controlled'] = out.Pressure_Ratio <= CONTROLLED_RATIO
    # Physics-based inputs: orifice flow scales with opening area x sqrt(pressure drop)
    out['dP_choke_psi'] = out.WHP_psig - out.Pdown_psig
    out['Flow_index'] = np.sqrt(out.dP_choke_psi) * out.Choke_pct / 100
    out['Restart'] = out.groupby('Well').Restart_count.diff().fillna(0) > 0
    out = add_segments(out).drop(columns=['Restart_count']).reset_index(drop=True)
    events = out[out.Well_event]

    out.to_csv(output_path, index=False)
    for label, n in log:
        print(f"  {label:<40s} {n:>6d}")
    print(f"Saved {len(out)} records from {out.Well.nunique()} wells to '{output_path}'.")
    print(f"  choke-controlled rows (ratio <= {CONTROLLED_RATIO}): {out.Choke_controlled.sum()}")
    print(f"  well events (restarts: {events.Restart.sum()}, water-cut steps: {(~events.Restart).sum()}): "
          + ", ".join(f"{w} {n}" for w, n in events.groupby('Well').size().items()))
    return out

if __name__ == "__main__":
    prepare_volve()
