"""
transferencia_externa.py — Transferencia de los modelos finales a fuentes externas
Entrena XGBoost y MLP (FEATURES_OPTIMAL_B e hiperparametros Optuna) sobre propio
y mendeley (familias base32 y word_based) y los evalua sobre la familia encoded
de Kaggle, no vista en el entrenamiento. Mide AUC y score medio de ataque y
benigno en cada destino.

Salida: resultados/transferencia_externa/transferencia_externa.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.modelo_xgboost import XGBFocused
from lib.modelo_mlp import MLPWrapper
from lib.constantes import (FEATURES_OPTIMAL as FEATURES,
                            XGB_BEST_HP as XGB_PARAMS,
                            MLP_BEST_HP as MLP_PARAMS)

ROOT    = Path(__file__).resolve().parents[2]
DATASET = ROOT / "Datasets" / "dataset_unificado.csv"
OUT_DIR = ROOT / "resultados" / "transferencia_externa"

MODELS = {
    'XGBoost': lambda: XGBFocused(**XGB_PARAMS),
    'MLP':     lambda: MLPWrapper(**MLP_PARAMS),
}

TRAIN_SOURCES = {'propio', 'mendeley', 'mendeley_benign'}

TARGETS = {
    'Kaggle_encoded': {
        'kaggle_cobaltstrike', 'kaggle_dns2tcp', 'kaggle_dnscat2',
        'kaggle_dnspot', 'kaggle_ozymandns', 'kaggle_tcp_over_dns',
        'kaggle_benign',
    },
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET, low_memory=False)
    train_df = df[df['source'].isin(TRAIN_SOURCES)].copy()
    X_tr = train_df[FEATURES].values.astype(np.float32)
    y_tr = train_df['attack'].values.astype(np.int32)

    rows = []
    for model_name, build in MODELS.items():
        m = build()
        m.fit_with_calibration(X_tr, y_tr)

        for tgt_name, tgt_sources in TARGETS.items():
            tgt_df = df[df['source'].isin(tgt_sources)].copy()
            if len(tgt_df) == 0:
                continue
            X_te = tgt_df[FEATURES].values.astype(np.float32)
            y_te = tgt_df['attack'].values.astype(np.int32)
            p_te = m.predict_proba(X_te)[:, 1]

            atk_scores = p_te[y_te == 1]
            ben_scores = p_te[y_te == 0]

            auc = float(roc_auc_score(y_te, p_te)) if y_te.sum() > 0 and (y_te == 0).sum() > 0 else float('nan')
            row = {
                'model':          model_name,
                'target':         tgt_name,
                'n_atk':          int(y_te.sum()),
                'n_ben':          int((y_te == 0).sum()),
                'auc':            round(auc, 4),
                'score_atk_mean': float(atk_scores.mean()) if len(atk_scores) else float('nan'),
                'score_atk_p10':  float(np.percentile(atk_scores, 10)) if len(atk_scores) else float('nan'),
                'score_atk_p90':  float(np.percentile(atk_scores, 90)) if len(atk_scores) else float('nan'),
                'score_ben_mean': float(ben_scores.mean()) if len(ben_scores) else float('nan'),
            }
            rows.append(row)
            print(f"{model_name}  {tgt_name}: AUC={auc:.4f}  "
                  f"atk_mean={row['score_atk_mean']:.4f}  ben_mean={row['score_ben_mean']:.4f}")

    out_csv = OUT_DIR / 'transferencia_externa.csv'
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Guardado: {out_csv}")


if __name__ == '__main__':
    main()
