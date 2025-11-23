from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


def get_models():
    models = {
        'RandomForest': RandomForestRegressor( n_estimators=100,
        max_depth=15,  # Limit depth to prevent memorization
        min_samples_split=10,  # Require more samples to split
        min_samples_leaf=5,  # Require more samples in leaves
        max_features='sqrt',  # Use subset of features for each split
        oob_score=True),
        'XGBoost': XGBRegressor(n_estimators=100, verbosity=0),
        'LightGBM': LGBMRegressor(n_estimators=100)
    }
    return models

