"""
feature_saturation.py — Curva de saturacion del numero de features
Parte de las 4 features del modelo final (orden del forward selection) y
extiende con greedy hasta 13, midiendo en cada k el AUC LODO por familia y el
AUC/AP in-domain (split 70/30). El punto k=4 es el modelo del trabajo.

Salida: resultados/feature_saturation/feature_saturation.csv
        (k, added, word_based, base32, encoded, lodo_mean, indomain_auc, indomain_ap)
Dataset: dataset_unificado.csv (3 familias)   Seed: 42
"""
import sys, time, logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.modelo_xgboost import XGBFocused
from lib.familias import assign_family, lodo_auc
from lib.constantes import (FEATURES_13 as CANDIDATES,
                            FEATURES_OPTIMAL as SELECTED_PREFIX,
                            XGB_DEFAULT_HP as XGB_DEFAULT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parents[2]
DATASET = ROOT / "Datasets" / "dataset_unificado.csv"
OUT_DIR = ROOT / "resultados" / "feature_saturation"
SEED = 42


def indomain_eval(df, features):
    X = df[features].values.astype(np.float32)
    y = df["attack"].values.astype(np.int32)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30,
                                          random_state=SEED, stratify=y)
    m = XGBFocused(**XGB_DEFAULT)
    m.fit_with_calibration(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(yte, p)), float(average_precision_score(yte, p))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.disable(logging.WARNING)

    df = pd.read_csv(DATASET, low_memory=False)
    df = assign_family(df)
    ben_all = df[df["family"] == "benign"].copy()
    log.info(f"Dataset {len(df):,} filas. Saturacion greedy sobre {len(CANDIDATES)} features.")

    selected, rows = [], []
    remaining = [f for f in CANDIDATES if f not in SELECTED_PREFIX]
    prev = -1.0
    t_start = time.time()

    for k in range(1, len(CANDIDATES) + 1):
        if k <= len(SELECTED_PREFIX):
            # Prefijo fijo: anade la siguiente feature del modelo final.
            best_feat = SELECTED_PREFIX[k - 1]
            best_detail = lodo_auc(df, selected + [best_feat], lambda: XGBFocused(**XGB_DEFAULT), ben_all)
            best_mean = best_detail["mean"]
        else:
            # Extension greedy: la mejor feature restante.
            best_feat, best_mean, best_detail = None, -1.0, None
            for feat in remaining:
                res = lodo_auc(df, selected + [feat], lambda: XGBFocused(**XGB_DEFAULT), ben_all)
                if res["mean"] > best_mean:
                    best_mean, best_feat, best_detail = res["mean"], feat, res
            remaining.remove(best_feat)
        selected.append(best_feat)
        auc_id, ap_id = indomain_eval(df, selected)
        rows.append({
            "k": k, "added": best_feat,
            "word_based": best_detail["word_based"],
            "base32": best_detail["base32"],
            "encoded": best_detail["encoded"],
            "lodo_mean": best_mean,
            "indomain_auc": auc_id, "indomain_ap": ap_id,
        })
        delta = best_mean - prev if prev >= 0 else float("nan")
        prev = best_mean
        log.info(f"k={k:2d}  +{best_feat:22s}  LODO={best_mean:.4f} (d={delta:+.4f})  "
                 f"in-domain AUC={auc_id:.4f}  [{time.time()-t_start:.0f}s]")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "feature_saturation.csv", index=False)
    log.info(f"\nGuardado: {OUT_DIR / 'feature_saturation.csv'}")
    kbest = int(out.loc[out['lodo_mean'].idxmax(), 'k'])
    log.info(f"k con maximo LODO medio: {kbest}  (LODO={out['lodo_mean'].max():.4f})")
    log.info(f"Tiempo total: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
