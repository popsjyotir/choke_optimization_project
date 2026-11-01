import pandas as pd
import joblib
from scipy.optimize import minimize_scalar

CRITICAL_THRESHOLD = 0.55

def find_optimal_choke(target_q_liquid, current_thp, current_p_line, current_gor, current_wc, model_path="models/choke_model.pkl"):
    """Calculates optimal choke opening size to achieve target liquid flow rate."""
    model = joblib.load(model_path)
    
    ratio = current_p_line / current_thp
    critical_flag = 1 if ratio <= CRITICAL_THRESHOLD else 0
    
    if critical_flag == 1:
        print(f"-> Flow Regime: CRITICAL (Sonic Flow). Pressure ratio = {ratio:.2f} <= {CRITICAL_THRESHOLD}")
    else:
        print(f"-> Flow Regime: SUBCRITICAL. Pressure ratio = {ratio:.2f} > {CRITICAL_THRESHOLD}")

    def objective(choke_size):
        input_data = pd.DataFrame([[
            current_thp, current_p_line, ratio, critical_flag, choke_size, current_gor, current_wc
        ]], columns=['THP_psi', 'P_line_psi', 'Pressure_Ratio', 'Is_Critical', 'Choke_64ths', 'GOR_scf_stb', 'Water_Cut'])
        predicted_q = model.predict(input_data)[0]
        return abs(predicted_q - target_q_liquid)

    res = minimize_scalar(objective, bounds=(16, 64), method='bounded')
    return res.x

if __name__ == "__main__":
    req_choke = find_optimal_choke(
        target_q_liquid=900,
        current_thp=1000,
        current_p_line=450,
        current_gor=800,
        current_wc=0.20
    )
    print(f"Recommended Choke Opening: {req_choke:.1f} / 64 inches")
