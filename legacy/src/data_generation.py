import numpy as np
import pandas as pd

def generate_choke_data(n_samples=1200, random_state=42):
    """Generates synthetic wellhead choke operational data."""
    np.random.seed(random_state)
    
    thp = np.random.uniform(400, 1400, n_samples)          # Tubing Head Pressure (psi)
    p_line = np.random.uniform(150, 900, n_samples)        # Flowline Pressure (psi)
    choke_64ths = np.random.uniform(16, 64, n_samples)     # Choke size (64ths)
    gor = np.random.uniform(300, 1500, n_samples)          # GOR (scf/STB)
    water_cut = np.random.uniform(0.05, 0.60, n_samples)   # Water Cut

    pressure_ratio = p_line / thp
    CRITICAL_THRESHOLD = 0.55
    is_critical = (pressure_ratio <= CRITICAL_THRESHOLD).astype(int)

    # Physics-informed flow calculation
    A, B, C = 12.0, 0.54, 1.89
    q_liquid = np.where(
        is_critical == 1,
        (thp * (choke_64ths ** C)) / (A * (gor ** B)),
        ((thp - p_line) * (choke_64ths ** C)) / (A * (gor ** B))
    )

    # Add measurement noise (+/- 7%)
    q_liquid_measured = q_liquid * np.random.normal(1, 0.07, n_samples)
    
    df = pd.DataFrame({
        'THP_psi': thp,
        'P_line_psi': p_line,
        'Pressure_Ratio': pressure_ratio,
        'Is_Critical': is_critical,
        'Choke_64ths': choke_64ths,
        'GOR_scf_stb': gor,
        'Water_Cut': water_cut,
        'Q_liquid_STB': q_liquid_measured
    })
    return df

if __name__ == "__main__":
    df = generate_choke_data()
    df.to_csv("data/choke_production_data.csv", index=False)
    print("Dataset generated and saved successfully as 'data/choke_production_data.csv'.")
