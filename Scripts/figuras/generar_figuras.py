"""
Genera las figuras del TFG a partir de los CSV de resultados
Lee los CSV escritos por entrenar_xgboost.py y entrenar_mlp.py y produce los PNG

Salida en TFG/img/:
  fig_seleccion_features.png   Progresion AUC LODO en el forward selection
  fig_feature_saturation.png    Curva de saturacion LODO vs in-domain (justifica 4 features)

Las figuras ROC/PR y de calibracion las genera generar_figuras_modelos.py 
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT  = Path(__file__).resolve().parents[2]
RES   = ROOT / "resultados"
IMG   = ROOT / "TFG" / "img"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.estilo_figuras import aplicar_estilo, C_XGB, C_MLP, C_BEN, C_FAM
aplicar_estilo()


def fig_seleccion_features():
    df = pd.read_csv(RES / "seleccion_features" / "seleccion_features.csv")
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    x = df["step"].values
    ax.plot(x, df["lodo_mean"], "o-", color=C_XGB, label="LODO medio", lw=2)
    ax.plot(x, df["word_based"], "s--", color=C_FAM["word_based"], label="word_based", lw=1.4, alpha=0.85)
    ax.plot(x, df["base32"], "v--", color=C_FAM["base32"], label="base32", lw=1.4, alpha=0.85)
    ax.plot(x, df["encoded"], "^--", color=C_FAM["encoded"], label="encoded", lw=1.4, alpha=0.85)
    for i, (s, v, feat) in enumerate(zip(df["step"], df["lodo_mean"], df["added"])):
        ax.annotate(feat, xy=(s, v), xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=9, color="black")
    ax.set_xlabel("Paso del forward selection")
    ax.set_ylabel("AUC LODO")
    ax.set_xticks(x)
    ax.set_ylim(0.86, 0.99)
    ax.legend(loc="lower right", frameon=False, ncol=2)
    plt.tight_layout()
    out = IMG / "fig_seleccion_features.png"
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"  guardada {out.name}")


def fig_feature_saturation():
    """Curva de saturacion: AUC LODO (generalizacion) e in-domain vs numero de features."""
    path = RES / "feature_saturation" / "feature_saturation.csv"
    if not path.exists():
        print("  (omitida fig_feature_saturation: falta feature_saturation.csv)")
        return
    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.plot(df["k"], df["indomain_auc"], "s--", color=C_BEN, lw=1.8,
            label="AUC in-domain (70/30)")
    ax.plot(df["k"], df["lodo_mean"], "o-", color=C_XGB, lw=2.2,
            label="AUC LODO medio (generalización)")
    # Banda de meseta del LODO (k=4..7)
    plat = df[(df["k"] >= 4) & (df["k"] <= 7)]["lodo_mean"]
    ax.axhspan(plat.min(), plat.max(), color=C_XGB, alpha=0.12, lw=0,
               label="meseta LODO (variación ≈ 0,003)")
    ax.axvline(4, color=C_MLP, lw=1.2, ls=":")
    ax.annotate("modelo final\n(4 features)", xy=(4, df.loc[df.k == 4, "lodo_mean"].iloc[0]),
                xytext=(6.0, 0.905), fontsize=9, color=C_MLP,
                arrowprops=dict(arrowstyle="->", color=C_MLP, lw=1.0))
    ax.set_xlabel("Número de features (orden greedy del forward selection)")
    ax.set_ylabel("AUC")
    ax.set_xticks(df["k"])
    ax.set_ylim(0.88, 1.01)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    plt.tight_layout()
    out = IMG / "fig_feature_saturation.png"
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"  guardada {out.name}")


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    fig_seleccion_features()
    fig_feature_saturation()
