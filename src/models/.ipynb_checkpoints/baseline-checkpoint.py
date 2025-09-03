from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

@dataclass
class SkModelWrapper:
    name: str
    model: Any

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

def make_model(name: str, params: Dict) -> Optional[SkModelWrapper]:
    name = name.lower()
    if name == "ridge":
        return SkModelWrapper(name, Ridge(**params))
    if name == "random_forest":
        return SkModelWrapper(name, RandomForestRegressor(n_jobs=-1, **params))
    if name == "xgboost":
        if not HAS_XGB:
            return None
        return SkModelWrapper(name, XGBRegressor(n_jobs=-1, tree_method="hist", objective="reg:squarederror", **params))
    raise ValueError(f"Unknown classical model: {name}")
