"""
XGBFocused: XGBoost con scale_pos_weight con tope, umbral de Youden y Platt
scaling ajustados en un cal-set interno, y feature importances para SHAP.
Comparte fit_with_calibration con el MLP para comparar de forma directa.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
import xgboost as xgb_lib


class PlattCalibrator:
    """Platt scaling: regresion logistica sobre los log-odds del modelo base."""

    def __init__(self, C: float = 1.0):
        self._lr = LogisticRegression(C=C, solver="lbfgs", max_iter=2000)

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> "PlattCalibrator":
        log_odds = self._to_log_odds(y_prob).reshape(-1, 1)
        self._lr.fit(log_odds, y_true)
        return self

    def predict_proba(self, y_prob: np.ndarray) -> np.ndarray:
        log_odds = self._to_log_odds(y_prob).reshape(-1, 1)
        return self._lr.predict_proba(log_odds)[:, 1]

    @staticmethod
    def _to_log_odds(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))


class XGBFocused:
    """XGBoost con umbral de Youden y Platt scaling ajustados en el cal-set."""

    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.7,
        min_child_weight: int = 5,
        reg_lambda: float = 2.0,
        gamma: float = 0.05,
        spw_cap: float = 15.0,
        seed: int = 42,
    ):
        self.seed    = seed
        self._hp     = dict(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample,
            colsample_bytree=colsample_bytree, min_child_weight=min_child_weight,
            reg_lambda=reg_lambda, gamma=gamma,
            eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0,
        )
        self._spw_cap = spw_cap
        self.clf_     = None
        self.threshold = 0.5
        self._platt   = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBFocused":
        """Entrena con scale_pos_weight con tope (umbral=0.5)."""
        n_neg, n_pos = int((y == 0).sum()), int((y == 1).sum())
        spw = min(n_neg / max(n_pos, 1), self._spw_cap)
        self.clf_ = xgb_lib.XGBClassifier(scale_pos_weight=spw, **self._hp)
        self.clf_.fit(X, y)
        return self

    def fit_with_calibration(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cal_size: float = 0.15,
    ) -> "XGBFocused":
        """Entrena en el 85% y ajusta umbral y Platt en el cal-set (15%)."""
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X, y, test_size=cal_size, stratify=y, random_state=self.seed)
        self.fit(X_fit, y_fit)
        prob_cal = self.predict_proba(X_cal)[:, 1]
        self.calibrate(y_cal, prob_cal)
        return self

    def calibrate(self, y_cal: np.ndarray, prob_cal: np.ndarray) -> "XGBFocused":
        """Ajusta el umbral de Youden y el Platt scaling sobre el cal-set."""
        try:
            fpr, tpr, thr = roc_curve(y_cal, prob_cal)
            self.threshold = float(thr[np.argmax(tpr - fpr)])
        except Exception:
            self.threshold = 0.5
        self._platt = PlattCalibrator().fit(y_cal, prob_cal)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probabilidades crudas del XGBoost."""
        p1 = self.clf_.predict_proba(X)[:, 1]
        return np.column_stack([1 - p1, p1])

    def predict_proba_platt(self, X: np.ndarray) -> np.ndarray:
        """Probabilidades recalibradas con Platt scaling."""
        if self._platt is None:
            return self.predict_proba(X)
        p1 = self._platt.predict_proba(self.predict_proba(X)[:, 1])
        return np.column_stack([1 - p1, p1])
