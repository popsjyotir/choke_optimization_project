import numpy as np
from src.models.ml import VOLVE_FEATURES, predict_hybrid
from src.models.empirical import choke_pct_to_64ths
from src.models.perkins import predict_perkins

def perkins_reference(df):
    """Perkins liquid rate at a 1-inch fully-open reference bore and Cd = 0.826 (scaled later by a fitted factor)."""
    q, r_c, is_crit = predict_perkins(df.WHP_psig, df.Pdown_psig, df.WHT_F, df.GOR_scf_stb, df.Water_Cut,
                                      choke_pct_to_64ths(df.Choke_pct, 64.0))
    return q, r_c, is_crit

def make_predictors(perkins_factor, hybrid=None, gradient_boosting=None):
    """Returns {name: predict(df) -> liquid rate} for the choke models used in what-if studies."""
    def perkins(df):
        return perkins_reference(df)[0] * perkins_factor

    predictors = {'Perkins': perkins}
    if hybrid is not None:
        predictors['Hybrid (Perkins + GB)'] = lambda df: predict_hybrid(hybrid, df[VOLVE_FEATURES], perkins(df))
    if gradient_boosting is not None:
        predictors['Gradient Boosting'] = lambda df: np.asarray(gradient_boosting.predict(df[VOLVE_FEATURES]))
    return predictors
