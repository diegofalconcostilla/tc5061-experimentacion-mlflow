# Actividad 3 — Código de experimentación base con MLflow (individual)

**Curso:** TC5061.10 · **Entrega:** domingo 11 de octubre de 2026, 23:59
**Entregable:** `entrega/TC5061_Experimentacion_MLFlow_FalconDiego.pdf`

El modelo base de la Actividad 2 (precio de diamantes, dataset Diamonds de Kaggle) se migró del notebook a un script de entrenamiento parametrizado, `src/train.py`, instrumentado con MLflow Tracking. El notebook se conserva sólo para la exploración de datos.

## Estructura

```
tc5061-experimentacion-mlflow/
├── src/train.py                    # script de entrenamiento (carga, limpieza, split, modelo, evaluación, MLflow)
├── run_experiments.ps1             # las 10 corridas del experimento (un hiperparámetro a la vez)
├── notebooks/exploracion_diamonds.ipynb   # sólo EDA y justificación de la limpieza
├── data/raw/diamonds.csv           # dataset original (Kaggle)
├── requirements.txt                # dependencias del script, versiones fijadas
├── requirements-notebook.txt       # dependencias adicionales del notebook
├── evidencias/                     # capturas de la interfaz de MLflow
├── entrega/                        # PDF entregable y su fuente HTML
└── mlflow.db, mlartifacts/         # tracking local (no versionado)
```

## Instalación

Requiere Python 3.12.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Ejecución

Todos los comandos se ejecutan desde la raíz del repositorio.

```powershell
# Corrida con los valores por defecto (Random Forest, n_estimators=200, max_depth=16,
# min_samples_leaf=2, max_features=0.5, seed=42)
.venv\Scripts\python src/train.py

# Cambiar hiperparámetros sin editar el código
.venv\Scripts\python src/train.py --n-estimators 200 --max-depth 16 --min-samples-leaf 2 --max-features 1.0 --seed 42

# Reproducir el baseline lineal de la Actividad 2
.venv\Scripts\python src/train.py --model linear

# Ver todos los argumentos con su descripción y valor por defecto
.venv\Scripts\python src/train.py --help

# Ejecutar las 10 corridas del experimento
.\run_experiments.ps1

# Abrir la interfaz de MLflow en http://localhost:5000
.venv\Scripts\mlflow ui --backend-store-uri sqlite:///mlflow.db
```

| Argumento | Defecto | Descripción |
|---|---|---|
| `--model` | `random_forest` | `random_forest` o `linear` (baseline de la Actividad 2) |
| `--n-estimators` | 200 | número de árboles |
| `--max-depth` | 16 | profundidad máxima; 0 = sin límite |
| `--min-samples-leaf` | 2 | mínimo de registros por hoja |
| `--max-features` | 0.5 | fracción de variables evaluadas en cada división |
| `--log-target` | apagado | entrena sobre log(price) |
| `--test-size` | 0.2 | proporción del conjunto de prueba |
| `--seed` | 42 | semilla aleatoria |
| `--experiment-name` | `TC5061-diamonds-precio-experimentacion` | experimento de MLflow |
| `--tracking-uri` | `sqlite:///<repositorio>/mlflow.db` | backend de tracking |
| `--run-name` | generado de los hiperparámetros | nombre de la corrida |

## Qué registra cada corrida

- Parámetros (`log_param`): tipo de modelo, hiperparámetros, semilla, tamaño de prueba, tamaños de train/test y variables.
- Métricas (`log_metric`): RMSE, MAE y R² en train y test, porcentaje de predicciones negativas, brecha de RMSE test–train y tiempo de entrenamiento.
- Artefactos (`log_artifact`): `metrics.json`, `graficas/diagnostico_residuos.png`, `graficas/importancia_permutacion.png` y `limpieza/bitacora_limpieza.txt`.
- Modelo (`mlflow.sklearn.log_model`): el pipeline completo (preprocesamiento + regresor) con firma y ejemplo de entrada.

## Reproducibilidad

La semilla (`--seed`) se fija en `random` y `numpy` y se pasa como `random_state` a la división train/test, al Random Forest y a la importancia por permutación. Con los mismos argumentos el resultado es idéntico: las corridas `rf-d16-l2-mf0.5-n200` y `rf-d16-l2-mf0.5-n200-rep` tienen exactamente las mismas métricas, y al borrar `mlflow.db` y volver a ejecutar `run_experiments.ps1` se obtienen los mismos valores. La única métrica que cambia entre ejecuciones es `tiempo_entrenamiento_s`, porque mide tiempo de reloj. Las versiones de todas las librerías están fijadas en `requirements.txt`.

## Resultados

| Corrida | RMSE test | MAE test | R² test |
|---|---|---|---|
| **rf-d16-l2-mf1-n200 (mejor)** | **543** | **279** | **0.981** |
| rf-d16-l2-mf0.5-n200 (defecto) | 554 | 285 | 0.981 |
| linear (baseline Actividad 2) | 1,180 | 808 | 0.912 |

El hiperparámetro más influyente fue `max_depth`. El análisis completo está en el PDF.
