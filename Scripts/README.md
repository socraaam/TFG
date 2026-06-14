# Scripts — Detección de DNS tunneling

Scripts usados 
 

## Scripts

### captura/

| Script | Función | Sección |
|---|---|---|
| `orquestador.sh` | Coordina las sesiones de exfiltración | §3.1 |
| `servidor_dns.sh` / `cliente_dns.sh` | Servidor (Kali) y cliente (Ubuntu) respectivamente. | §3.1 |
| `pcap_a_dataset.py` | Extrae el dataset propio desde los PCAP con el esquema de Mendeley | §3.1 |

### dataset/

| Script | Función | Sección |
|---|---|---|
| `construir_dataset_unificado.py` | Construye `dataset_unificado.csv` | §3.2 |

### features/ 

| Script | Función | Sección |
|---|---|---|
| `seleccion_features.py` | Produce el conjunto final de 4 features | §3.3 |
| `feature_saturation.py` | Curva de saturación para justificar las 4 features | §3.3 |

### modelos/

| Script | Función | Sección |
|---|---|---|
| `entrenar_xgboost.py` | Optimización, entrenamiento y resultados | §3.4, cap. 4 |
| `entrenar_mlp.py` | Optimización, entrenamiento y resultados | §3.4, cap. 4 |

### evaluacion/ 

| Script | Función | Sección |
|---|---|---|
| `transferencia_externa.py` | Experimento de transferencia | cap. 4 |

### lib/ 

| Script | Función |
|---|---|
| `modelo_xgboost.py` |
| `modelo_mlp.py` |
| `familias.py` | 

### figuras/ 

| Script | Salida |
|---|---|
| `generar_figuras.py` | Figuras estáticas
| `generar_figuras_modelos.py` | Figuras que requieren reentrenar XGB y MLP 


## Asistencia de IA

El código de `Scripts/` se escribió con ayuda de Claude Sonnet 4.6 (Anthropic)
en tareas de revisión y supervisión.
