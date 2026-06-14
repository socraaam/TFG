"""
familias.py — Asignacion de familia y evaluacion LODO compartidas
Centraliza el mapa fuente->familia, la politica de benignos y el bucle
leave-one-family-out (LODO) usados por seleccion_features.py,
entrenar_xgboost.py, entrenar_mlp.py y feature_saturation.py.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

FAMILY_MAP = {
    'mendeley':            'word_based',
    'kaggle_dnsexfil_mod': 'word_based',
    'propio':              'base32',
    'kaggle_iodine':       'base32',
    'kaggle_dnsexfil':     'base32',
    'mendeley_encoded':    'base32',
    'kaggle_cobaltstrike': 'encoded',
    'kaggle_dns2tcp':      'encoded',
    'kaggle_dnscat2':      'encoded',
    'kaggle_dnspot':       'encoded',
    'kaggle_ozymandns':    'encoded',
    'kaggle_tcp_over_dns': 'encoded',
    'mendeley_benign':     'benign',
    'kaggle_benign':       'benign',
}

FAMILIES = ['word_based', 'base32', 'encoded']


def assign_family(df: pd.DataFrame, propio_as_benign: bool = True) -> pd.DataFrame:
    """Anade la columna 'family' a partir de 'source'.

    Con propio_as_benign=True los benignos de la fuente propia se reetiquetan
    como 'benign' (politica de los modelos finales y de la saturacion).
    seleccion_features.py usa False para reproducir TAB:FWD_SEL.
    """
    df = df.copy()
    df['family'] = df['source'].map(FAMILY_MAP).fillna('benign')
    if propio_as_benign:
        df.loc[(df['source'] == 'propio') & (df['attack'] == 0), 'family'] = 'benign'
    return df


def lodo_auc(df: pd.DataFrame, features: list, make_model, ben_all: pd.DataFrame = None) -> dict:
    """Por cada familia entrena con las otras dos mas los benignos y evalua
    sobre la familia retenida mas los benignos. Devuelve {familia: AUC,
    'mean': media}."""
    if ben_all is None:
        ben_all = df[df['family'] == 'benign'].copy()
    res = {}
    for held in FAMILIES:
        train_df = df[df['family'] != held]
        test_df  = pd.concat([df[df['family'] == held], ben_all], ignore_index=True)
        if test_df['attack'].nunique() < 2:
            continue
        X_tr = train_df[features].values.astype(np.float32)
        y_tr = train_df['attack'].values.astype(np.int32)
        X_te = test_df[features].values.astype(np.float32)
        y_te = test_df['attack'].values.astype(np.int32)
        m = make_model()
        m.fit_with_calibration(X_tr, y_tr)
        p = m.predict_proba(X_te)[:, 1]
        res[held] = roc_auc_score(y_te, p)
    res['mean'] = float(np.mean([res[f] for f in FAMILIES if f in res]))
    return res


def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error con bins uniformes, ponderado por población."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)
