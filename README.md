# Detección de exfiltración de datos mediante DNS tunneling

Trabajo Fin de Grado (EPS-UAM). Detección de túneles DNS con aprendizaje
automático sobre el contenido estructural del FQDN, con foco en la
generalización entre familias de codificación (*cross-domain*).

## Resultados del modelo final

Sobre `Datasets/dataset_unificado.csv` (511 746 registros, familias base32,
word_based y encoded) con cuatro features estructurales del FQDN
(`avg_label_len`, `subdomain_len`, `special_ratio`, `fqdn_len`):

| Métrica | XGBoost (principal) | MLP (contraste) |
|---|---|---|
| AUC in-domain | 0,9987 | 0,9856 |
| AUC LODO medio | **0,9616** | 0,9091 |
| ECE (Platt) | 0,004 | 0,030 |
| Brier | 0,009 | 0,046 |

Una cuarta fuente candidata, CIC-Bell-DNS-EXF-2021, queda excluida del
entrenamiento, el motivo se detalla en la memoria.

## Estructura del repositorio

```
TFG/
├── Scripts/         (ver Scripts/README.md)
│   ├── captura/     Laboratorio de captura del dataset propio (§3.1)
│   ├── dataset/     Construcción del dataset unificado (§3.2)
│   ├── features/    Selección y saturación de features (§3.3)
│   ├── modelos/     HPO + entrenamiento final XGBoost y MLP (§3.4)
│   ├── evaluacion/  Transferencia entre dominios y auditoría CIC (cap. 4)
│   ├── figuras/     Generación de las figuras de la memoria
│   └── lib/         Código compartido (XGBFocused, familias, Platt)
├── Datasets/        dataset_ataque.csv y dataset_benigno.csv (capturas propias) y
│                    dataset_unificado.csv (dataset final, CIC excluido)
├── requirements.txt
└── README.md
```

## Reproducir los resultados

Requiere `dataset_unificado.csv` (incluido). Con la semilla fijada a 42 en
todas las etapas.

```powershell
# Cadena principal (modelo final)
python Scripts/features/seleccion_features.py        # selección de las 4 features
python Scripts/modelos/entrenar_xgboost.py            # XGBoost: HPO (50 trials) + evaluación
python Scripts/modelos/entrenar_mlp.py                # MLP: HPO (10 trials) + evaluación
python Scripts/features/feature_saturation.py         # curva de saturación (justifica las 4)
python Scripts/figuras/generar_figuras_modelos.py     # puntos de operación + ROC/PR/calibración/confusión
python Scripts/figuras/generar_figuras.py             # resto de figuras (desde resultados/)

# Experimentos complementarios
python Scripts/evaluacion/transferencia_externa.py    # transferencia a Kaggle encoded (no visto)
```

El paso previo de construcción del dataset
(`python -m Scripts.dataset.construir_dataset_unificado`) necesita además las
fuentes crudas de Mendeley, Kaggle y CIC, que por su tamaño **no se
distribuyen** con el repositorio, su ubicación se indica con la variable de
entorno `TFG_DATA_ROOT` (por defecto, la ruta del autor). Las capturas propias
(`dataset_ataque.csv` y `dataset_benigno.csv`) sí se incluyen en `Datasets/`. A
partir de `dataset_unificado.csv`, que también se incluye, el resto de pasos
se reproduce sin acceso a los datos crudos.

Las salidas alimentan las tablas y figuras de la memoria y se guardan en
`resultados/modelo_xgboost/`, `resultados/modelo_mlp/`,
`resultados/seleccion_features/`, `resultados/feature_saturation/`,
`resultados/operating_points/` y `resultados/transferencia_externa/`.

## Dependencias

```
pip install -r requirements.txt
```

Versiones verificadas: Python 3.11.7, NumPy 2.4.4, pandas 3.0.2,
scikit-learn 1.8.0, XGBoost 3.2.0, Optuna 4.8.0, SHAP 0.42+, LIME 0.2.0.1,
SciPy 1.17.1, matplotlib 3.10.9, seaborn 0.13.2, scapy 2.5+. El MLP usa
`sklearn.MLPClassifier`.
