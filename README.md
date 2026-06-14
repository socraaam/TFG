# Detección de exfiltración de datos usando DNS tunneling

Trabajo Fin de Grado (EPS-UAM). Detección de túneles DNS con aprendizaje
automático sobre el contenido estructural del FQDN, con foco en la
generalización entre familias de codificación.

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
│                    dataset_unificado.csv (dataset final)
└── README.md
```

## Reproducir los resultados

```cmd
# Cadena principal (modelo final)
python Scripts/features/seleccion_features.py        # selección de las 4 features
python Scripts/modelos/entrenar_xgboost.py            # XGBoost: HPO (50 trials) + evaluación
python Scripts/modelos/entrenar_mlp.py                # MLP: HPO (10 trials) + evaluación
python Scripts/features/feature_saturation.py         # curva de saturación 
python Scripts/figuras/generar_figuras_modelos.py     # puntos de operación + ROC/PR/calibración/confusión
python Scripts/figuras/generar_figuras.py             # resto de figuras 

# Experimentos complementarios
python Scripts/evaluacion/transferencia_externa.py    # transferencia a Kaggle encoded 
```

Los resultados se obtienen de las salidas de los programas

