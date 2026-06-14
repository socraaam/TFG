"""
entrenar_xgboost.py — HPO Optuna + entrenamiento final XGBoost + evaluacion
HPO propio (Optuna, 50 trials, LODO_mean) y, con los mejores hiperparametros,
entrenamiento y evaluacion del modelo final.

Features : FEATURES_OPTIMAL_B = [avg_label_len, subdomain_len, special_ratio, fqdn_len]
Dataset  : dataset_unificado.csv (3 familias)
Explicab.: SHAP TreeExplainer sobre 2000 ejemplos del test

Guarda en resultados/modelo_xgboost/:
  modelo_xgboost.ubj                  — modelo serializado
  modelo_xgboost_metadatos.json       — features, params, metricas
  modelo_xgboost_lodo_familias.csv    — LODO por familia
  modelo_xgboost_dominio.csv          — metricas in-domain
  modelo_xgboost_shap.csv             — importancia SHAP
"""
import sys, json, logging
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import shap
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.modelo_xgboost import XGBFocused
from lib.familias import assign_family, lodo_auc, compute_ece, FAMILIES
from lib.constantes import FEATURES_OPTIMAL as FEATURES

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parents[2]
DATASET  = ROOT / "Datasets" / "dataset_unificado.csv"
OUT_DIR  = ROOT / "resultados" / "modelo_xgboost"
SEED     = 42
N_TRIALS = 50

OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATASET, low_memory=False)
df = assign_family(df)
log.info(f"Dataset {len(df):,} filas: {df['family'].value_counts().to_dict()}")

ben_all = df[df['family'] == 'benign'].copy()

# 1. HPO Optuna (LODO_mean sobre el dataset completo)
log.info(f"HPO Optuna ({N_TRIALS} trials)...")


def objective(trial):
    params = dict(
        n_estimators      = trial.suggest_int('n_estimators', 100, 1000),
        max_depth         = trial.suggest_int('max_depth', 3, 8),
        learning_rate     = trial.suggest_float('learning_rate', 0.01, 0.30, log=True),
        subsample         = trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree  = trial.suggest_float('colsample_bytree', 0.5, 1.0),
        min_child_weight  = trial.suggest_int('min_child_weight', 1, 20),
        reg_lambda        = trial.suggest_float('reg_lambda', 0.5, 10.0, log=True),
        gamma             = trial.suggest_float('gamma', 0.0, 0.5),
        spw_cap           = trial.suggest_float('spw_cap', 1.0, 4.0),
        seed              = SEED,
    )
    return lodo_auc(df, FEATURES, lambda: XGBFocused(**params), ben_all)['mean']


sampler = optuna.samplers.TPESampler(seed=SEED)
study   = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=N_TRIALS)

best = study.best_trial
log.info(f"Mejor trial {best.number}: LODO_mean={best.value:.4f}")

BEST_PARAMS = dict(**best.params, seed=SEED)

# 2. LODO por familia con los mejores hiperparametros
lodo_rows = []
for held_fam in FAMILIES:
    train_df  = df[df['family'] != held_fam].copy()
    atk_test  = df[df['family'] == held_fam].copy()
    test_df   = pd.concat([atk_test, ben_all], ignore_index=True)

    X_tr = train_df[FEATURES].values.astype(np.float32)
    y_tr = train_df['attack'].values.astype(np.int32)
    X_te = test_df[FEATURES].values.astype(np.float32)
    y_te = test_df['attack'].values.astype(np.int32)

    m = XGBFocused(**BEST_PARAMS)
    m.fit_with_calibration(X_tr, y_tr)
    p_te = m.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, p_te)
    ap  = average_precision_score(y_te, p_te)
    atk_scores = p_te[y_te == 1]
    log.info(f"  {held_fam}: AUC={auc:.4f}  AP={ap:.4f}")
    lodo_rows.append({'family': held_fam, 'auc': auc, 'ap': ap,
                      'n_atk': int(y_te.sum()), 'n_ben': int((y_te==0).sum()),
                      'score_atk_mean': float(atk_scores.mean()),
                      'score_atk_p10': float(np.percentile(atk_scores, 10)),
                      'score_atk_p90': float(np.percentile(atk_scores, 90))})

lodo_df = pd.DataFrame(lodo_rows)
lodo_mean = float(lodo_df['auc'].mean())
log.info(f"LODO_mean = {lodo_mean:.4f}")
lodo_df.to_csv(OUT_DIR / 'modelo_xgboost_lodo_familias.csv', index=False)

# 3. In-domain (split 70/30 estratificado)
X_all = df[FEATURES].values.astype(np.float32)
y_all = df['attack'].values.astype(np.int32)
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.30,
                                            random_state=SEED, stratify=y_all)
m_id = XGBFocused(**BEST_PARAMS)
m_id.fit_with_calibration(X_tr, y_tr)
p_id = m_id.predict_proba(X_te)[:, 1]
auc_id  = roc_auc_score(y_te, p_id)
ap_id   = average_precision_score(y_te, p_id)
ece_raw = compute_ece(y_te, p_id)
brier   = float(np.mean((p_id - y_te)**2))

# El calibrador Platt se ajusta en el cal-set interno, nunca sobre el test.
p_platt = m_id.predict_proba_platt(X_te)[:, 1]
ece_platt = compute_ece(y_te, p_platt)

atk_sc = p_platt[y_te==1]; ben_sc = p_platt[y_te==0]
log.info(f"In-domain: AUC={auc_id:.4f}  AP={ap_id:.4f}  ECE_Platt={ece_platt:.4f}  Brier={brier:.4f}")

indomain = {'auc': auc_id, 'ap': ap_id, 'ece_raw': ece_raw,
            'ece_platt': ece_platt, 'brier': brier,
            'score_atk_mean': float(atk_sc.mean()), 'score_ben_mean': float(ben_sc.mean())}
pd.DataFrame([indomain]).to_csv(OUT_DIR / 'modelo_xgboost_dominio.csv', index=False)

# 4. SHAP (reusa el modelo in-domain)
explainer = shap.TreeExplainer(m_id.clf_)
rng = np.random.default_rng(SEED)
idx = rng.choice(len(X_te), size=min(2000, len(X_te)), replace=False)
sv = explainer.shap_values(X_te[idx])
mean_abs = np.abs(sv).mean(axis=0)
shap_df = pd.DataFrame({'feature': FEATURES, 'mean_abs_shap': mean_abs})
shap_df = shap_df.sort_values('mean_abs_shap', ascending=False)
shap_df.to_csv(OUT_DIR / 'modelo_xgboost_shap.csv', index=False)

# 5. Modelo final sobre todo el dataset
m_final = XGBFocused(**BEST_PARAMS)
m_final.fit_with_calibration(X_all, y_all)
m_final.clf_.save_model(str(OUT_DIR / 'modelo_xgboost.ubj'))

metadata = {
    'features': FEATURES,
    'best_params': BEST_PARAMS,
    'hpo_lodo_mean': best.value,
    'lodo_mean_final': lodo_mean,
    'lodo_by_family': {r['family']: r['auc'] for r in lodo_rows},
    'indomain': indomain,
    'n_trials': N_TRIALS,
    'hpo_full_dataset': True,
    'dataset': str(DATASET),
    'n_samples': len(df),
}
with open(OUT_DIR / 'modelo_xgboost_metadatos.json', 'w') as f:
    json.dump(metadata, f, indent=2)

log.info(f"Guardado en {OUT_DIR}  (LODO_mean={lodo_mean:.4f})")
