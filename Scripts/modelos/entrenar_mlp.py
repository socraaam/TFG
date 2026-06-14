"""
entrenar_mlp.py — HPO Optuna + entrenamiento final MLP + evaluacion
Features: FEATURES_OPTIMAL_B = [avg_label_len, subdomain_len, special_ratio, fqdn_len]
Dataset: dataset_unificado.csv (3 familias)
Modelo: sklearn MLPClassifier + Platt scaling (comparable con XGBoost)
HPO: Optuna 10 trials, LODO_mean, dataset completo
Explicabilidad: LIME sobre 300 ejemplos del test

Guarda en resultados/modelo_mlp/:
  modelo_mlp.pkl                  — modelo + scaler (joblib)
  modelo_mlp_metadatos.json       — features, params, metricas
  modelo_mlp_lodo_familias.csv    — LODO por familia
  modelo_mlp_dominio.csv          — metricas in-domain
  modelo_mlp_lime.csv             — importancia LIME por feature
"""
import sys, json, logging
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from joblib import dump as joblib_dump

sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.familias import assign_family, lodo_auc, compute_ece, FAMILIES
from lib.modelo_mlp import MLPWrapper
from lib.constantes import FEATURES_OPTIMAL as FEATURES

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parents[2]
DATASET  = ROOT / "Datasets" / "dataset_unificado.csv"
OUT_DIR  = ROOT / "resultados" / "modelo_mlp"
SEED     = 42
N_TRIALS = 10
HPO_MAX_ITER           = 50    # cap durante HPO
HPO_N_ITER_NO_CHANGE   = 5     # early stopping agresivo en HPO
FINAL_MAX_ITER         = 300   # cap modelo final
FINAL_N_ITER_NO_CHANGE = 20    # early stopping normal en final


def params_from_trial(trial, max_iter, n_iter_no_change):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    units    = [trial.suggest_int(f'n_units_l{i}', 32, 128) for i in range(n_layers)]
    return dict(
        hidden_layer_sizes  = tuple(units),
        activation          = trial.suggest_categorical('activation', ['relu', 'tanh']),
        alpha               = trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
        lr_init             = trial.suggest_float('lr_init', 1e-4, 5e-2, log=True),
        batch_size          = trial.suggest_categorical('batch_size', [256, 512]),
        max_iter            = max_iter,
        n_iter_no_change    = n_iter_no_change,
    )


OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATASET, low_memory=False)
df = assign_family(df)
log.info(f"Dataset {len(df):,} filas: {df['family'].value_counts().to_dict()}")

ben_all = df[df['family'] == 'benign'].copy()

# 1. HPO Optuna (LODO_mean sobre el dataset completo)
log.info(f"HPO Optuna ({N_TRIALS} trials)...")


def objective(trial):
    p = params_from_trial(trial, HPO_MAX_ITER, HPO_N_ITER_NO_CHANGE)
    return lodo_auc(df, FEATURES, lambda: MLPWrapper(**p), ben_all)['mean']


sampler = optuna.samplers.TPESampler(seed=SEED)
study   = optuna.create_study(direction='maximize', sampler=sampler)
study.optimize(objective, n_trials=N_TRIALS)

best = study.best_trial
log.info(f"Mejor trial {best.number}: LODO_mean={best.value:.4f}")

# Hiperparametros finales (max_iter completo)
best_p = dict(best.params)
n_layers = best_p.pop('n_layers')
units = tuple(best_p.pop(f'n_units_l{i}') for i in range(n_layers))
BEST_PARAMS = dict(
    hidden_layer_sizes  = units,
    activation          = best_p['activation'],
    alpha               = best_p['alpha'],
    lr_init             = best_p['lr_init'],
    batch_size          = best_p['batch_size'],
    max_iter            = FINAL_MAX_ITER,
    n_iter_no_change    = FINAL_N_ITER_NO_CHANGE,
)

# 2. LODO por familia con los mejores hiperparametros
lodo_rows = []
for held_fam in FAMILIES:
    tr_df  = df[df['family'] != held_fam].copy()
    atk_te = df[df['family'] == held_fam].copy()
    te_df  = pd.concat([atk_te, ben_all], ignore_index=True)

    X_tr = tr_df[FEATURES].values.astype(np.float32)
    y_tr = tr_df['attack'].values.astype(np.int32)
    X_te = te_df[FEATURES].values.astype(np.float32)
    y_te = te_df['attack'].values.astype(np.int32)

    m = MLPWrapper(**BEST_PARAMS)
    m.fit_with_calibration(X_tr, y_tr)
    p_te = m.predict_proba(X_te)[:, 1]
    auc  = roc_auc_score(y_te, p_te)
    ap   = average_precision_score(y_te, p_te)
    atk_scores = p_te[y_te == 1]
    log.info(f"  {held_fam}: AUC={auc:.4f}  AP={ap:.4f}")
    lodo_rows.append({'family': held_fam, 'auc': auc, 'ap': ap,
                      'n_atk': int(y_te.sum()), 'n_ben': int((y_te==0).sum()),
                      'score_atk_mean': float(atk_scores.mean()),
                      'score_atk_p10':  float(np.percentile(atk_scores, 10)),
                      'score_atk_p90':  float(np.percentile(atk_scores, 90)),
                      'n_iter': m.n_iter_})

lodo_df = pd.DataFrame(lodo_rows)
lodo_mean = float(lodo_df['auc'].mean())
log.info(f"LODO_mean = {lodo_mean:.4f}")
lodo_df.to_csv(OUT_DIR / 'modelo_mlp_lodo_familias.csv', index=False)

# 3. In-domain (split 70/30 estratificado)
X_all = df[FEATURES].values.astype(np.float32)
y_all = df['attack'].values.astype(np.int32)
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.30,
                                            random_state=SEED, stratify=y_all)
m_id = MLPWrapper(**BEST_PARAMS)
m_id.fit_with_calibration(X_tr, y_tr)
p_id   = m_id.predict_proba(X_te)[:, 1]
auc_id = roc_auc_score(y_te, p_id)
ap_id  = average_precision_score(y_te, p_id)
brier  = float(np.mean((p_id - y_te) ** 2))

raw_id = m_id.clf_.predict_proba(m_id.scaler_.transform(X_te))[:, 1]
ece_raw   = compute_ece(y_te, raw_id)
ece_platt = compute_ece(y_te, p_id)

atk_sc = p_id[y_te == 1]
ben_sc = p_id[y_te == 0]
log.info(f"In-domain: AUC={auc_id:.4f}  AP={ap_id:.4f}  ECE={ece_raw:.4f}  Brier={brier:.4f}")

indomain = {'auc': auc_id, 'ap': ap_id, 'ece_raw': ece_raw,
            'ece_platt': ece_platt, 'brier': brier,
            'score_atk_mean': float(atk_sc.mean()), 'score_ben_mean': float(ben_sc.mean()),
            'n_iter': m_id.n_iter_, 'threshold': m_id.threshold}
pd.DataFrame([indomain]).to_csv(OUT_DIR / 'modelo_mlp_dominio.csv', index=False)

# 4. LIME
explainer = LimeTabularExplainer(
    X_tr.astype(np.float64),
    feature_names=FEATURES,
    class_names=['benign', 'attack'],
    mode='classification',
    random_state=SEED,
    discretize_continuous=False,
)
rng = np.random.default_rng(SEED)
n_lime = min(300, len(X_te))
idx = rng.choice(len(X_te), size=n_lime, replace=False)
X_lime = X_te[idx].astype(np.float64)

lime_weights = {f: [] for f in FEATURES}
for xi in X_lime:
    exp = explainer.explain_instance(
        xi,
        lambda x: m_id.predict_proba(x.astype(np.float32)),
        num_features=len(FEATURES),
        num_samples=500,
    )
    for feat, w in exp.as_list():
        # extrae el nombre base de la feature ("avg_label_len <= 3.50" -> avg_label_len)
        for fname in FEATURES:
            if fname in feat:
                lime_weights[fname].append(abs(w))
                break

lime_df = pd.DataFrame({
    'feature': FEATURES,
    'mean_abs_lime': [np.mean(lime_weights[f]) for f in FEATURES],
}).sort_values('mean_abs_lime', ascending=False)
lime_df.to_csv(OUT_DIR / 'modelo_mlp_lime.csv', index=False)

# 5. Modelo final sobre todo el dataset
m_final = MLPWrapper(**BEST_PARAMS)
m_final.fit_with_calibration(X_all, y_all)
joblib_dump(m_final, OUT_DIR / 'modelo_mlp.pkl')

metadata = {
    'features': FEATURES,
    'best_params': {k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in BEST_PARAMS.items()},
    'hpo_lodo_mean': best.value,
    'lodo_mean_final': lodo_mean,
    'lodo_by_family': {r['family']: r['auc'] for r in lodo_rows},
    'indomain': indomain,
    'n_trials': N_TRIALS,
    'hpo_full_dataset': True,
    'dataset': str(DATASET),
    'n_samples': len(df),
}
with open(OUT_DIR / 'modelo_mlp_metadatos.json', 'w') as f:
    json.dump(metadata, f, indent=2)

log.info(f"Guardado en {OUT_DIR}  (LODO_mean={lodo_mean:.4f})")
