# Scripts — Detección de DNS tunneling

Scripts usados en la
memoria del TFG organizados por etapa 

## Estructura

```
Scripts/
├── captura/      # Laboratorio de captura del dataset propio (§3.1)
├── dataset/      # Construcción del dataset unificado (§3.2)
├── features/     # Selección y saturación de features (§3.3)
├── modelos/      # Entrenamiento final XGBoost y MLP (§3.4)
├── evaluacion/   # Domain shift y auditoría CIC (cap. 4)
├── figuras/      # Generación de las figuras de la memoria
├── lib/          # Código compartido (XGBFocused, familias, Platt)
└── README.md     # Este fichero
```

Orden de ejecución: `captura` → `dataset` → `features` → `modelos` → `figuras`
(`evaluacion` es complementario). 

## Datasets de entrada (no están en GitHub por tamaño)

Las rutas se resuelven con la variable de entorno `TFG_DATA_ROOT` (por defecto,
`C:/Users/Marcos/Desktop/Compartida_ubuntu/TFG/`).

| Dataset | Subcarpeta | Origen | Uso |
|---|---|---|---|
| Mendeley | `dataset_mendeley/` | [data.mendeley.com c4n7fckkz3 v3](https://data.mendeley.com/datasets/c4n7fckkz3/3) | base32 + word\_based + benignos |
| Kaggle | `dataset_kaggle/` | [kaggle.com daumel dns-tunneling-dataset](https://www.kaggle.com/datasets/daumel/dns-tunneling-dataset) | 9 herramientas + benignos |
| CIC-Bell-DNS-EXF-2021 | `dataset_cic/` | [unb.ca/cic/datasets/dns-exf-2021](https://www.unb.ca/cic/datasets/dns-exf-2021.html) | Excluido (§4.2) |

Las capturas propias del laboratorio (`dataset_out/`) ya extraídas con
`pcap_a_dataset.py` se incluyen directamente en `Datasets/` (ver tabla
siguiente), así que no hace falta repetir esa extracción.

## Datasets incluidos en `Datasets/`

| Fichero | Filas | Composición |
|---|---|---|
| `dataset_ataque.csv` | 27\,240 | Capturas propias del laboratorio (§3.1), tráfico de túnel iodine (familia base32). |
| `dataset_benigno.csv` | 31\,953 | Capturas propias del laboratorio (§3.1), tráfico DNS benigno. |
| `dataset_unificado.csv` | 511\,746 | Dataset final: tres familias (word\_based, base32, encoded) más benignos. Sin CIC. |

## Scripts

### captura/ — laboratorio del dataset propio

| Script | Función | Sección |
|---|---|---|
| `orquestador.sh` | Coordina las sesiones de exfiltración | §3.1 |
| `servidor_dns.sh` / `cliente_dns.sh` | Roles servidor (Kali) y cliente (Ubuntu) del túnel iodine que invoca el orquestador. | §3.1 |
| `pcap_a_dataset.py` | Extrae el dataset propio desde los PCAP con formato de columnas compatible con Mendeley. | §3.1 |

### dataset/ — construcción del dataset

| Script | Función | Sección |
|---|---|---|
| `construir_dataset_unificado.py` | Construye `dataset_unificado.csv` desde las tres procedencias. Implementa los caps por familia, el muestreo estratificado de Mendeley y la extracción desde PCAP. | §3.2 |

### features/ — selección de features

| Script | Función | Sección |
|---|---|---|
| `seleccion_features.py` | Forward selection greedy sobre las 13 candidatas con criterio LODO medio. Produce el conjunto final de 4 features. | §3.3 |
| `feature_saturation.py` | Curva de saturación LODO vs in-domain de k=4 a 13 features. Justifica el conjunto de 4. | §3.3 |

### modelos/ — entrenamiento final

| Script | Función | Sección |
|---|---|---|
| `entrenar_xgboost.py` | Script autocontenido del XGBoost: HPO Optuna de 50 trials + entrenamiento final. Calcula in-domain, LODO por familia y SHAP. | §3.4, cap. 4 |
| `entrenar_mlp.py` | Script autocontenido del MLP: HPO Optuna de 10 trials + entrenamiento final. Calcula in-domain, LODO por familia y LIME. | §3.4, cap. 4 |

### evaluacion/ — domain shift y auditoría CIC

| Script | Función | Sección |
|---|---|---|
| `transferencia_externa.py` | Entrena el XGBoost y el MLP finales sobre propio+Mendeley y mide su transferencia a la familia `encoded` de Kaggle, no vista en el entrenamiento. | cap. 4 |

### lib/ — componentes reutilizables

| Script | Función |
|---|---|
| `modelo_xgboost.py` | Clase `XGBFocused` con interfaz `fit_with_calibration` (encapsula `scale_pos_weight` con tope, umbral de Youden y feature importances para SHAP) y la clase `PlattCalibrator` (regresión logística sobre log-odds). |
| `modelo_mlp.py` | Clase `MLPWrapper` (sklearn `MLPClassifier` con escalado y calibración Platt) usada en la comparativa XGBoost vs MLP. |
| `familias.py` | `FAMILY_MAP`, `assign_family`, el bucle LODO (`lodo_auc`) y `compute_ece` compartidos por `seleccion_features`, `entrenar_xgboost`, `entrenar_mlp` y `feature_saturation`. |

### figuras/ — figuras de la memoria

| Script | Salida |
|---|---|
| `generar_figuras.py` | Figuras estáticas (forward selection y curva de saturación) en `TFG_LaTeX/img/`. Lee únicamente CSVs ya escritos en `resultados/`; no reentrena nada. |
| `generar_figuras_modelos.py` | Figuras que requieren reentrenar XGB y MLP sobre el split 70/30: ROC/PR, calibración y matrices de confusión, más `resultados/operating_points/operating_points.csv`. |

## Reproducción completa

```powershell
# Desde la raíz del proyecto
python -m Scripts.dataset.construir_dataset_unificado   # 5 min, requiere los 3 datasets en disco
python Scripts/features/seleccion_features.py           # 10 min
python Scripts/modelos/entrenar_xgboost.py              # ~40 min (HPO Optuna + entrenamiento final)
python Scripts/modelos/entrenar_mlp.py                  # ~30 min (HPO Optuna + entrenamiento final)
python Scripts/features/feature_saturation.py           # ~15 min (curva de saturación, justifica las 4 features)
python Scripts/figuras/generar_figuras_modelos.py       # ~3 min (reentrena XGB y MLP: ROC/PR, calibración, confusión)
python Scripts/figuras/generar_figuras.py               # < 30 s (figuras estáticas desde CSV)
```

Experimentos complementarios citados en la memoria (no son parte de la cadena
principal):

```powershell
python Scripts/evaluacion/transferencia_externa.py   # transferencia a Kaggle encoded, no visto (cap. 4)
```

Semilla 42 en todos los scripts. Todos los CSV de salida son
deterministas frente a las mismas entradas y la misma semilla.

## Versiones verificadas

Python 3.11.7 · numpy 2.4.4 · pandas 3.0.2 · scikit-learn 1.8.0 ·
xgboost 3.2.0 · optuna 4.8.0 · shap 0.51.0 · lime 0.2.0.1 ·
scipy 1.17.1 · matplotlib 3.10.9 · scapy 2.7.0.

La extracción del dataset propio desde los PCAP (`captura/pcap_a_dataset.py`)
necesita además `tldextract`.

## Asistencia de IA

El código de `Scripts/` se escribió con ayuda de Claude Sonnet 4.6 (Anthropic)
en tareas de revisión y supervisión.
