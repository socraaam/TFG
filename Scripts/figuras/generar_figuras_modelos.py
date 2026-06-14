"""
generar_figuras_modelos.py — Metricas operativas y figuras de resultados de modelos
Reentrena el XGBoost y el MLP finales sobre el mismo split in-domain 70/30
(semilla 42) y produce:

  resultados/operating_points/operating_points.csv
      Punto de operacion de cada modelo en el umbral de Youden: TPR, FPR,
      precision, recall, F1, especificidad, accuracy, MCC, FPR@TPR=0.95 y
      precision@TPR=0.95.

  TFG_LaTeX/img/fig_roc_pr.png      Curvas ROC y precision-recall (XGB vs MLP).
  TFG_LaTeX/img/fig_confusion.png   Matrices de confusion in-domain (umbral Youden).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_curve, confusion_matrix,
                             matthews_corrcoef)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Scripts"))
from lib.modelo_xgboost import XGBFocused
from lib.modelo_mlp import MLPWrapper
from lib.constantes import (FEATURES_OPTIMAL as FEATURES,
                            XGB_BEST_HP as XGB_PARAMS,
                            MLP_BEST_HP as MLP_PARAMS)
from lib.estilo_figuras import aplicar_estilo, C_XGB, C_MLP

RES = ROOT / "resultados"
IMG = ROOT / "TFG_LaTeX" / "img"
OP_DIR = RES / "operating_points"
SEED = 42

aplicar_estilo()


def operating_metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    spec = tn / (tn + fp)
    acc = (tp + tn) / len(y)
    f1 = 2 * prec * tpr / (prec + tpr) if (prec + tpr) else 0.0
    mcc = matthews_corrcoef(y, yhat)
    # FPR @ TPR=0.95 y precision @ TPR=0.95 (independientes del umbral)
    fprc, tprc, _ = roc_curve(y, p)
    fpr_at_95 = float(np.interp(0.95, tprc, fprc))
    prc, rec, _ = precision_recall_curve(y, p)
    # precision_recall_curve devuelve recall decreciente; ordenar ascendente
    order = np.argsort(rec)
    prec_at_95 = float(np.interp(0.95, rec[order], prc[order]))
    return dict(threshold=thr, tpr=tpr, fpr=fpr, precision=prec, recall=tpr,
                f1=f1, specificity=spec, accuracy=acc, mcc=mcc,
                fpr_at_tpr95=fpr_at_95, precision_at_tpr95=prec_at_95,
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn))


def main():
    OP_DIR.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ROOT / "Datasets" / "dataset_unificado.csv", low_memory=False)
    X = df[FEATURES].values.astype(np.float32)
    y = df['attack'].values.astype(np.int32)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30,
                                          random_state=SEED, stratify=y)

    print("Entrenando XGBoost...")
    xgb = XGBFocused(**XGB_PARAMS).fit_with_calibration(Xtr, ytr)
    p_xgb = xgb.predict_proba(Xte)[:, 1]
    print("Entrenando MLP...")
    mlp = MLPWrapper(**MLP_PARAMS).fit_with_calibration(Xtr, ytr)
    p_mlp = mlp.predict_proba(Xte)[:, 1]

    # Vectores de test cuya combinacion de features no aparece en el train
    ktr = {tuple(np.round(v, 6)) for v in Xtr}
    novel = np.array([tuple(np.round(v, 6)) not in ktr for v in Xte])
    print(f"  Vectores de test novedosos: {int(novel.sum())} ({100*novel.mean():.1f}%)")

    rows = []
    for name, p, thr in [("XGBoost", p_xgb, xgb.threshold),
                         ("MLP", p_mlp, mlp.threshold)]:
        m = operating_metrics(yte, p, thr)
        m['model'] = name
        m['auc'] = float(roc_auc_score(yte, p))
        m['ap'] = float(average_precision_score(yte, p))
        if novel.sum() > 0 and len(np.unique(yte[novel])) == 2:
            m['auc_novel'] = float(roc_auc_score(yte[novel], p[novel]))
            m['ap_novel'] = float(average_precision_score(yte[novel], p[novel]))
        else:
            m['auc_novel'] = float('nan'); m['ap_novel'] = float('nan')
        m['n_novel'] = int(novel.sum())
        rows.append(m)
        print(f"  {name}: thr={thr:.3f} TPR={m['tpr']:.4f} FPR={m['fpr']:.4f} "
              f"prec={m['precision']:.4f} F1={m['f1']:.4f} MCC={m['mcc']:.4f} "
              f"FPR@95TPR={m['fpr_at_tpr95']:.4f}  AUC_novel={m['auc_novel']:.4f}")
    cols = ['model', 'auc', 'ap', 'auc_novel', 'ap_novel', 'n_novel',
            'threshold', 'tpr', 'fpr', 'precision',
            'recall', 'f1', 'specificity', 'accuracy', 'mcc',
            'fpr_at_tpr95', 'precision_at_tpr95', 'tp', 'fp', 'tn', 'fn']
    op = pd.DataFrame(rows)[cols]
    op.to_csv(OP_DIR / "operating_points.csv", index=False)
    print(f"Guardado: {OP_DIR / 'operating_points.csv'}")

    # Figura ROC + PR
    fig, (axr, axp) = plt.subplots(1, 2, figsize=(8.4, 3.8))
    for name, p, c, thr in [("XGBoost", p_xgb, C_XGB, xgb.threshold),
                            ("MLP", p_mlp, C_MLP, mlp.threshold)]:
        fpr, tpr, _ = roc_curve(yte, p)
        auc = roc_auc_score(yte, p)
        axr.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC={auc:.4f})")
        # punto de operacion
        yhat = (p >= thr)
        op_fpr = yhat[yte == 0].mean(); op_tpr = yhat[yte == 1].mean()
        axr.plot(op_fpr, op_tpr, 'o', color=c, ms=7, mec='white', mew=1.0)
        prc, rec, _ = precision_recall_curve(yte, p)
        ap = average_precision_score(yte, p)
        axp.plot(rec, prc, color=c, lw=2, label=f"{name} (AP={ap:.4f})")
    axr.plot([0, 1], [0, 1], color='gray', lw=0.8, ls=':')
    axr.set_xlim(-0.005, 0.16); axr.set_ylim(0.84, 1.005)
    axr.set_xlabel("FPR (tasa de falsos positivos)")
    axr.set_ylabel("TPR (tasa de detección)")
    axr.set_title("ROC (zoom esquina superior izquierda)", fontsize=10)
    axr.legend(loc="lower right", frameon=False, fontsize=9)
    axp.set_xlim(0.84, 1.005); axp.set_ylim(0.84, 1.005)
    axp.set_xlabel("Recall"); axp.set_ylabel("Precisión")
    axp.set_title("Precisión-Recall (zoom)", fontsize=10)
    axp.legend(loc="lower left", frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(IMG / "fig_roc_pr.png", bbox_inches="tight"); plt.close()
    print(f"  guardada fig_roc_pr.png")

    # Matriz de confusion in-domain (umbral de Youden); color = fraccion por fila
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7))
    labels = ["Benigno", "Ataque"]
    for ax, name, p, thr, cmap in [
            (axes[0], "XGBoost", p_xgb, xgb.threshold, "Blues"),
            (axes[1], "MLP", p_mlp, mlp.threshold, "Reds")]:
        yhat = (p >= thr).astype(int)
        cm = confusion_matrix(yte, yhat)              # [[TN, FP], [FN, TP]]
        cm_row = cm / cm.sum(axis=1, keepdims=True)   # normalizado por fila real
        ax.imshow(cm_row, cmap=cmap, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                txt = f"{cm[i, j]:,}\n({cm_row[i, j]*100:.1f}\\%)".replace("\\%", "%")
                ax.text(j, i, txt, ha="center", va="center",
                        color="white" if cm_row[i, j] > 0.5 else "black",
                        fontsize=10)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
        ax.set_title(name, fontsize=11)
        ax.grid(False)
        ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
        ax.tick_params(which="minor", length=0)
    plt.tight_layout()
    plt.savefig(IMG / "fig_confusion.png", bbox_inches="tight"); plt.close()
    print(f"  guardada fig_confusion.png")


if __name__ == "__main__":
    main()
