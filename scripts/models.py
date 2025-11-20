from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


def get_models():
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=80, oob_score=True, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=100, verbosity=0),
        'LightGBM': LGBMRegressor(n_estimators=100)
    }
    return models

