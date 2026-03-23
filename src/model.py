import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

class TriBoostEnsemble(BaseEstimator, RegressorMixin):
    """
    A unified Scikit-Learn wrapper that ensembles the predictions
    of three tree-based regressors via mean averaging.
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=self.random_state)
        self.lgb_model = lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=self.random_state, verbose=-1)
        self.rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=self.random_state)
        
    def fit(self, X, y):
        # Fit independent models
        self.xgb_model.fit(X, y)
        self.lgb_model.fit(X, y)
        self.rf_model.fit(X, y)
        return self
        
    def predict(self, X):
        # Ensemble inference via mean prediction
        pred_xgb = self.xgb_model.predict(X)
        pred_lgb = self.lgb_model.predict(X)
        pred_rf = self.rf_model.predict(X)
        return np.mean([pred_xgb, pred_lgb, pred_rf], axis=0)
