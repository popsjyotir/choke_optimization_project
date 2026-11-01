import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

def train_choke_model(data_path="data/choke_production_data.csv", model_output_path="models/choke_model.pkl"):
    """Trains a Random Forest surrogate model on choke performance data."""
    df = pd.read_csv(data_path)
    
    X = df[['THP_psi', 'P_line_psi', 'Pressure_Ratio', 'Is_Critical', 'Choke_64ths', 'GOR_scf_stb', 'Water_Cut']]
    y = df['Q_liquid_STB']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print("Model Training Complete!")
    print(f"-> R² Score: {r2:.3f}")
    print(f"-> RMSE: {rmse:.2f} STB/day")

    joblib.dump(model, model_output_path)
    print(f"Trained model saved to '{model_output_path}'.")

if __name__ == "__main__":
    train_choke_model()
