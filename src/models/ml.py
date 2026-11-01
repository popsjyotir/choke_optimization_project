import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VOLVE_FEATURES = ['WHP_psig', 'Pdown_psig', 'Pressure_Ratio', 'Choke_pct', 'WHT_F', 'GOR_scf_stb', 'Water_Cut',
                  'dP_choke_psi', 'Flow_index']
DABIRI_FEATURES = ['Pwh_psi', 'GLR_scf_stb', 'Choke_64ths']

# Physical direction of each input's effect on rate (+1 rises, -1 falls), used by the monotonic model
MONOTONIC = {
    'WHP_psig': 1, 'Pdown_psig': -1, 'Pressure_Ratio': -1, 'Choke_pct': 1, 'dP_choke_psi': 1, 'Flow_index': 1,
    'log_q_test': 1, 'dChoke_since_test': 1, 'dPdown_since_test': -1,
    'Pwh_psi': 1, 'GLR_scf_stb': -1, 'Choke_64ths': 1,
}

# Small grids: the goal is a fair, tuned model, not a leaderboard search
MODELS = {
    'Random Forest': (
        lambda cols: RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=1),  # grid search parallelizes
        {'regressor__min_samples_leaf': [1, 5, 20], 'regressor__max_features': [0.5, 1.0]},
    ),
    'Gradient Boosting': (
        lambda cols: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=42),
        {'regressor__max_leaf_nodes': [15, 31], 'regressor__l2_regularization': [0.0, 1.0],
         'regressor__loss': ['squared_error', 'absolute_error']},
    ),
    'GB (monotonic)': (
        lambda cols: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=42,
                                                   monotonic_cst=[MONOTONIC.get(c, 0) for c in cols]),
        {'regressor__max_leaf_nodes': [15, 31], 'regressor__l2_regularization': [0.0, 1.0]},
    ),
    'MLP': (
        lambda cols: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), early_stopping=True,
                                                                  max_iter=2000, random_state=42)),
        {'regressor__mlpregressor__alpha': [1e-4, 1e-2],
         'regressor__mlpregressor__hidden_layer_sizes': [(64, 64), (128, 64)]},
    ),
}

def _log_target(estimator):
    """Trains on log(rate) so errors are relative, matching the AAPRE metric."""
    return TransformedTargetRegressor(regressor=estimator, func=np.log, inverse_func=np.exp)

def fit_ml(name, X, y, groups=None):
    """Tunes and fits an ML model. With groups (wells or time blocks), tuning folds never share a group."""
    make, grid = MODELS[name]
    if groups is not None:
        cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(_log_target(make(list(X.columns))), grid, cv=cv,
                          scoring='neg_mean_absolute_percentage_error', n_jobs=-1)
    search.fit(X, y, groups=groups)
    return search.best_estimator_

def fit_hybrid(X, y, q_physics, groups=None):
    """Gradient boosting that learns the log-ratio between observed and physics-predicted rates."""
    Xh = X.assign(log_q_physics=np.log(q_physics))
    model = fit_ml('Gradient Boosting', Xh, y / q_physics, groups)
    return model

def predict_hybrid(model, X, q_physics):
    return q_physics * model.predict(X.assign(log_q_physics=np.log(q_physics)))

def fit_quantiles(X, y, quantiles=(0.1, 0.9)):
    """Gradient boosting quantile models (on log rate) for prediction intervals."""
    models = {}
    for q in quantiles:
        m = HistGradientBoostingRegressor(loss='quantile', quantile=q, max_iter=400, learning_rate=0.05,
                                          max_leaf_nodes=31, random_state=42)
        models[q] = _log_target(m).fit(X, y)
    return models
