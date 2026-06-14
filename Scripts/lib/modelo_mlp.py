"""
modelo_mlp.py — MLPWrapper para la comparativa XGBoost vs MLP
MLP de scikit-learn con StandardScaler + Platt scaling, con la misma interfaz
que XGBFocused (fit_with_calibration / predict_proba / threshold).
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.utils.class_weight import compute_sample_weight

SEED = 42


class MLPWrapper:
    """sklearn MLPClassifier con StandardScaler + Platt scaling."""

    def __init__(self, hidden_layer_sizes, activation, alpha,
                 lr_init, batch_size, max_iter, n_iter_no_change=20, seed=SEED):
        self.scaler_ = StandardScaler()
        self.clf_ = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            alpha=alpha,
            learning_rate_init=lr_init,
            batch_size=batch_size,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=n_iter_no_change,
            random_state=seed,
            solver='adam',
            verbose=False,
        )
        self._platt   = None
        self.threshold = 0.5
        self.n_iter_   = 0
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit_with_calibration(self, X, y, cal_size=0.15):
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X, y, test_size=cal_size, stratify=y, random_state=SEED)
        sw = compute_sample_weight('balanced', y_fit)
        Xs = self.scaler_.fit_transform(X_fit)
        self.clf_.fit(Xs, y_fit, sample_weight=sw)
        self.n_iter_ = self.clf_.n_iter_
        raw_cal = self.clf_.predict_proba(self.scaler_.transform(X_cal))[:, 1]
        self._platt = LogisticRegression(C=1.0)
        self._platt.fit(raw_cal.reshape(-1, 1), y_cal)
        p_platt = self._platt.predict_proba(raw_cal.reshape(-1, 1))[:, 1]
        try:
            fpr, tpr, thr = roc_curve(y_cal, p_platt)
            self.threshold = float(thr[np.argmax(tpr - fpr)])
        except Exception:
            self.threshold = 0.5
        return self

    def predict_proba(self, X):
        Xs = self.scaler_.transform(X)
        raw = self.clf_.predict_proba(Xs)[:, 1]
        if self._platt is not None:
            p1 = self._platt.predict_proba(raw.reshape(-1, 1))[:, 1]
        else:
            p1 = raw
        return np.column_stack([1 - p1, p1])
