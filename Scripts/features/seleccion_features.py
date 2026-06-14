"""
Greedy forward feature selection para el XGBoost
Dataset  : dataset_unificado.csv 
Metrica  : mejorar LODO_mean
Criterio : greedy forward, anadir feature mientras DELTA LODO_mean > 0.001
Seed     : 42

Usa assign_family(propio_as_benign=False), la politica con la que se obtuvo
TAB:FWD_SEL (los modelos finales usan propio_as_benign=True).
"""

import sys, time, logging
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.modelo_xgboost import XGBFocused
from lib.familias import assign_family, lodo_auc
from lib.constantes import FEATURES_13 as CANDIDATES, XGB_DEFAULT_HP as XGB_DEFAULT

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parents[2]
DATASET   = ROOT / "Datasets" / "dataset_unificado.csv"
OUT_DIR   = ROOT / "resultados" / "seleccion_features"
THRESHOLD = 0.001


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET, low_memory=False)
    df = assign_family(df, propio_as_benign=False)
    log.info(f"Dataset {len(df):,} filas. Forward selection sobre {len(CANDIDATES)} features.")

    selected = []
    remaining = list(CANDIDATES)
    history = []
    prev_best = 0.0
    step = 0

    while remaining:
        step += 1
        best_feat, best_mean, best_detail = None, prev_best, {}
        t0 = time.time()
        for feat in remaining:
            res = lodo_auc(df, selected + [feat], lambda: XGBFocused(**XGB_DEFAULT))
            if res["mean"] > best_mean:
                best_mean, best_feat, best_detail = res["mean"], feat, res
        delta = best_mean - prev_best

        if best_feat is None or delta < THRESHOLD:
            log.info(f"Fin: mejor delta={delta:.4f} < {THRESHOLD}")
            break

        selected.append(best_feat)
        remaining.remove(best_feat)
        prev_best = best_mean
        history.append({
            "step":       step,
            "added":      best_feat,
            "lodo_mean":  best_mean,
            "delta":      delta,
            "word_based": best_detail.get("word_based"),
            "base32":     best_detail.get("base32"),
            "encoded":    best_detail.get("encoded"),
            "elapsed_s":  round(time.time() - t0, 1),
        })
        log.info(f"k={step}  +{best_feat:22s}  LODO_mean={best_mean:.4f}  delta=+{delta:.4f}")

    df_hist = pd.DataFrame(history)
    df_hist.to_csv(OUT_DIR / "seleccion_features.csv", index=False)
    log.info(f"Guardado: {OUT_DIR / 'seleccion_features.csv'}")
    log.info(f"FEATURES_OPTIMAL_B = {selected}  (LODO_mean={prev_best:.4f})")

    return selected, prev_best, df_hist


if __name__ == "__main__":
    main()
