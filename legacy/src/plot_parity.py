import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def plot_parity(data_path="data/choke_production_data.csv", model_path="models/choke_model.pkl", output_path="outputs/choke_parity_comparison.png"):
    """Generates and saves a parity plot comparing Random Forest vs. Gilbert Equation."""
    df = pd.read_csv(data_path)
    
    X = df[['THP_psi', 'P_line_psi', 'Pressure_Ratio', 'Is_Critical', 'Choke_64ths', 'GOR_scf_stb', 'Water_Cut']]
    y = df['Q_liquid_STB']
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. Random Forest Predictions
    rf_model = joblib.load(model_path)
    y_pred_rf = rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, y_pred_rf)
    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    
    # 2. Classical Gilbert Equation Predictions
    A, B, C = 12.0, 0.54, 1.89
    thp_test = X_test['THP_psi']
    choke_test = X_test['Choke_64ths']
    gor_test = X_test['GOR_scf_stb']
    y_pred_gilbert = (thp_test * (choke_test ** C)) / (A * (gor_test ** B))
    gilbert_r2 = r2_score(y_test, y_pred_gilbert)
    gilbert_rmse = np.sqrt(mean_squared_error(y_test, y_pred_gilbert))
    
    # 3. Plotting Setup
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    min_val = min(y_test.min(), y_pred_rf.min(), y_pred_gilbert.min())
    max_val = max(y_test.max(), y_pred_rf.max(), y_pred_gilbert.max())
    
    # Random Forest Parity Plot
    axes[0].scatter(y_test, y_pred_rf, color='dodgerblue', alpha=0.6, edgecolors='k', linewidths=0.5)
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Ideal Line')
    axes[0].set_title(f'Random Forest Surrogate Model\n$R^2 = {rf_r2:.3f}$ | RMSE = {rf_rmse:.1f} STB/d', fontsize=11)
    axes[0].set_xlabel('Actual Liquid Rate (STB/day)', fontsize=10)
    axes[0].set_ylabel('Predicted Liquid Rate (STB/day)', fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='upper left')
    
    # Gilbert Equation Parity Plot
    axes[1].scatter(y_test, y_pred_gilbert, color='darkorange', alpha=0.6, edgecolors='k', linewidths=0.5)
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Ideal Line')
    axes[1].set_title(f'Classical Gilbert Equation\n$R^2 = {gilbert_r2:.3f}$ | RMSE = {gilbert_rmse:.1f} STB/d', fontsize=11)
    axes[1].set_xlabel('Actual Liquid Rate (STB/day)', fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(loc='upper left')
    
    plt.suptitle('Choke Performance Benchmark: Actual vs. Predicted', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Parity plot saved successfully as '{output_path}'.")

if __name__ == "__main__":
    plot_parity()
