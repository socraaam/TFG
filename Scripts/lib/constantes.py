"""
constantes.py — Listas de features e hiperparametros compartidos.

  FEATURES_13      : 13 candidatas, en el orden en que
                     las produce compute_features() y las recorre el forward
                     selection.
  FEATURES_OPTIMAL : las 4 del modelo final, en el orden del forward selection.
  XGB_DEFAULT_HP   : hiperparametros por defecto del XGBoost (forward selection
                     y curva de saturacion).
  XGB_BEST_HP      : hiperparametros Optuna del XGBoost final.
  MLP_BEST_HP      : hiperparametros Optuna del MLP final.
"""

SEED = 42

FEATURES_13 = [
    "fqdn_len", "subdomain_len", "labels", "max_label",
    "entropy", "digits_ratio",
    "subdomain_fraction", "entropy_x_subdomain",
    "max_label_fraction", "avg_label_len",
    "special_ratio", "alpha_run_ratio",
    "vowel_ratio",
]

FEATURES_OPTIMAL = ["avg_label_len", "subdomain_len", "special_ratio", "fqdn_len"]

XGB_DEFAULT_HP = dict(
    n_estimators=250, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=5, reg_lambda=2.0, gamma=0.05,
    spw_cap=3.0, seed=SEED,
)

XGB_BEST_HP = dict(
    n_estimators=522, max_depth=5,
    learning_rate=0.012739412118882226,
    subsample=0.9231810640140996,
    colsample_bytree=0.9120517338863218,
    min_child_weight=8,
    reg_lambda=4.833164136038232,
    gamma=0.13878496455057732,
    spw_cap=3.781318326222462,
    seed=SEED,
)

MLP_BEST_HP = dict(
    hidden_layer_sizes=(111,),
    activation='tanh',
    alpha=0.01216413935141706,
    lr_init=0.0001584325068438869,
    batch_size=256,
    max_iter=300,
    n_iter_no_change=20,
)
