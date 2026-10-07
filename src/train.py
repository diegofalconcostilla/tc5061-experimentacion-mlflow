"""Entrenamiento parametrizado del modelo de precio de diamantes, instrumentado con MLflow Tracking.

Migra a un script el flujo del notebook baseline de la Actividad 2 (carga, limpieza, división,
entrenamiento y evaluación) y registra cada corrida en MLflow: parámetros, métricas, artefactos y modelo.

Uso (desde la carpeta de la actividad):
    python src/train.py                                  # valores por defecto
    python src/train.py --n-estimators 300 --max-depth 20 --min-samples-leaf 1 --seed 42
    python src/train.py --model linear                   # reproduce el baseline lineal de la Actividad 2
    python src/train.py --help                           # descripción de todos los argumentos
"""

import argparse
import json
import random
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: el script corre en terminal y sólo guarda figuras

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Carpeta de la actividad (el script vive en src/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Orden natural de las escalas de calidad del GIA, de peor a mejor
ORDEN = {
    "cut": ["Fair", "Good", "Very Good", "Premium", "Ideal"],
    "color": ["J", "I", "H", "G", "F", "E", "D"],
    "clarity": ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"],
}
FEATURES_NUM = ["carat", "depth", "table"]
FEATURES_CAT = ["cut", "color", "clarity"]
TARGET = "price"

# Umbral de inconsistencia geométrica (puntos de depth) definido en la limpieza del notebook
UMBRAL_DEPTH = 5.0

SKOPS_TRUSTED_TYPES = ["numpy.dtype", "sklearn.tree._tree.Tree", "numpy.log", "numpy.exp"]


# --------------------------------------------------------------------------------------------
# Argumentos de línea de comandos
# --------------------------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Entrena un modelo de regresión del precio de diamantes (Kaggle Diamonds) "
                    "y registra la corrida en MLflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    modelo = parser.add_argument_group("modelo e hiperparámetros")
    modelo.add_argument("--model", choices=["random_forest", "linear"], default="random_forest",
                        help="Algoritmo. 'linear' reproduce el baseline de la Actividad 2 e ignora los "
                             "hiperparámetros del bosque.")
    modelo.add_argument("--n-estimators", type=int, default=200,
                        help="Número de árboles del Random Forest.")
    modelo.add_argument("--max-depth", type=int, default=16,
                        help="Profundidad máxima de cada árbol; 0 significa sin límite.")
    modelo.add_argument("--min-samples-leaf", type=int, default=2,
                        help="Mínimo de registros en cada hoja; valores altos suavizan el modelo.")
    modelo.add_argument("--max-features", type=float, default=0.5,
                        help="Fracción de variables evaluadas en cada división (0 < valor <= 1).")
    modelo.add_argument("--log-target", action=argparse.BooleanOptionalAction, default=False,
                        help="Entrena sobre log(price) y regresa las predicciones a USD.")

    datos = parser.add_argument_group("datos y reproducibilidad")
    datos.add_argument("--test-size", type=float, default=0.2,
                       help="Proporción del conjunto de prueba.")
    datos.add_argument("--seed", type=int, default=42,
                       help="Semilla aleatoria: controla la división, el modelo y la importancia por permutación.")
    datos.add_argument("--data-path", type=Path, default=BASE_DIR / "data" / "raw" / "diamonds.csv",
                       help="Ruta del CSV original de Kaggle.")

    tracking = parser.add_argument_group("MLflow")
    tracking.add_argument("--experiment-name", default="TC5061-diamonds-precio-experimentacion",
                          help="Experimento de MLflow donde se agrupan las corridas.")
    tracking.add_argument("--tracking-uri", default=f"sqlite:///{(BASE_DIR / 'mlflow.db').as_posix()}",
                          help="Backend de tracking de MLflow.")
    tracking.add_argument("--run-name", default=None,
                          help="Nombre de la corrida; si se omite se genera a partir de los hiperparámetros.")

    args = parser.parse_args(argv)
    if not 0 < args.max_features <= 1:
        parser.error("--max-features debe estar en el intervalo (0, 1]")
    if not 0 < args.test_size < 1:
        parser.error("--test-size debe estar en el intervalo (0, 1)")
    return args


def set_seed(seed):
    """Fija la semilla de las fuentes de aleatoriedad globales (el resto recibe random_state explícito)."""
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------------------------
def load_data(path):
    """Carga el CSV original y convierte las escalas de calidad en categóricas ordenadas."""
    df = pd.read_csv(path)
    for col, niveles in ORDEN.items():
        df[col] = pd.Categorical(df[col], categories=niveles, ordered=True)
    return df


def clean_data(df):
    """Aplica las decisiones de limpieza del notebook y devuelve el dataset y una bitácora de pasos."""
    bitacora = [("Dataset original", len(df))]

    df = df.drop_duplicates()
    bitacora.append(("Sin duplicados exactos", len(df)))

    # Medidas en 0 mm: faltantes disfrazados
    df = df[(df[["x", "y", "z"]] > 0).all(axis=1)]
    bitacora.append(("Sin dimensiones en cero (x, y o z = 0)", len(df)))

    # depth reportado vs. calculado a partir de x, y, z: diferencias grandes son errores de captura
    depth_calc = df["z"] / ((df["x"] + df["y"]) / 2) * 100
    df = df[(depth_calc - df["depth"]).abs() <= UMBRAL_DEPTH]
    bitacora.append((f"Sin geometría inconsistente (depth > {UMBRAL_DEPTH:g} puntos)", len(df)))

    # x, y, z sólo sirven para limpiar: son colineales con carat y se excluyen del modelo
    df = df[FEATURES_NUM + FEATURES_CAT + [TARGET]].reset_index(drop=True)
    return df, bitacora


def split_data(df, test_size, seed):
    """División train/test estratificada por rangos de peso, la variable más predictiva."""
    estratos = pd.cut(df["carat"], bins=[0, 0.4, 0.7, 1.0, 1.5, np.inf], labels=False)
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=estratos)


# --------------------------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------------------------
def build_pipeline(args):
    """Pipeline completo: preprocesamiento + regresor, con transformación logarítmica opcional del objetivo."""
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), FEATURES_NUM),
        ("cat", OneHotEncoder(categories=[ORDEN[c] for c in FEATURES_CAT], handle_unknown="ignore"),
         FEATURES_CAT),
    ])

    if args.model == "linear":
        regresor = LinearRegression()
    else:
        regresor = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth or None,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            random_state=args.seed,
            n_jobs=-1,
        )

    if args.log_target:
        regresor = TransformedTargetRegressor(regressor=regresor, func=np.log, inverse_func=np.exp)

    return Pipeline([("preprocessor", preprocessor), ("model", regresor)])


def model_params(args):
    """Hiperparámetros que efectivamente usa el modelo elegido (los del bosque no aplican al lineal)."""
    params = {"model_type": args.model, "log_target": args.log_target}
    if args.model == "random_forest":
        params.update({
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth or "None",
            "min_samples_leaf": args.min_samples_leaf,
            "max_features": args.max_features,
        })
    return params


def default_run_name(args):
    nombre = "linear" if args.model == "linear" else (
        f"rf-d{args.max_depth or 'None'}-l{args.min_samples_leaf}-mf{args.max_features:g}-n{args.n_estimators}"
    )
    return nombre + ("-log" if args.log_target else "")


# --------------------------------------------------------------------------------------------
# Evaluación y artefactos
# --------------------------------------------------------------------------------------------
def evaluate(y_true, y_pred):
    """Métricas de regresión en USD más el porcentaje de predicciones imposibles (negativas)."""
    return {
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "pct_pred_negativas": float((y_pred < 0).mean()),
    }


def plot_diagnostics(y_true, y_pred, titulo):
    """Predicción vs. real y residuos vs. predicción sobre el conjunto de prueba."""
    residuos = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y_true, y_pred, alpha=0.15, s=5)
    axes[0].plot([0, 20_000], [0, 20_000], "r--", label="predicción perfecta")
    axes[0].set(xlabel="Precio real (USD)", ylabel="Precio predicho (USD)", title="Predicción vs. real (test)")
    axes[0].legend()
    axes[1].scatter(y_pred, residuos, alpha=0.15, s=5)
    axes[1].axhline(0, color="r", ls="--")
    axes[1].set(xlabel="Precio predicho (USD)", ylabel="Residuo (USD)", title="Residuos vs. predicción (test)")
    fig.suptitle(titulo)
    fig.tight_layout()
    return fig


def plot_permutation_importance(pipeline, X_test, y_test, seed, n_muestra=5_000):
    """Importancia por permutación sobre las variables originales (comparable entre algoritmos)."""
    muestra = X_test.sample(min(n_muestra, len(X_test)), random_state=seed)
    resultado = permutation_importance(pipeline, muestra, y_test.loc[muestra.index], n_repeats=5,
                                       random_state=seed, scoring="neg_root_mean_squared_error", n_jobs=1)
    importancias = pd.Series(resultado.importances_mean, index=muestra.columns).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    importancias.plot.barh(ax=ax, color="steelblue")
    ax.set(xlabel="Aumento del RMSE al permutar la variable (USD)",
           title="Importancia por permutación (muestra de test)")
    fig.tight_layout()
    return fig, importancias


# --------------------------------------------------------------------------------------------
# Orquestación
# --------------------------------------------------------------------------------------------
def setup_mlflow(tracking_uri, experiment_name):
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=(BASE_DIR / "mlartifacts").as_uri())
    mlflow.set_experiment(experiment_name)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)

    df_raw = load_data(args.data_path)
    df, bitacora = clean_data(df_raw)
    X_train, X_test, y_train, y_test = split_data(df, args.test_size, args.seed)

    pipeline = build_pipeline(args)
    inicio = time.perf_counter()
    pipeline.fit(X_train, y_train)
    tiempo_entrenamiento = time.perf_counter() - inicio

    pred_train = pipeline.predict(X_train)
    pred_test = pipeline.predict(X_test)
    metricas = {
        **{f"train_{k}": v for k, v in evaluate(y_train, pred_train).items() if k != "pct_pred_negativas"},
        **{f"test_{k}": v for k, v in evaluate(y_test, pred_test).items()},
    }
    metricas["brecha_rmse_test_train"] = metricas["test_rmse"] - metricas["train_rmse"]

    run_name = args.run_name or default_run_name(args)
    fig_diag = plot_diagnostics(y_test.to_numpy(), pred_test, run_name)
    fig_imp, importancias = plot_permutation_importance(pipeline, X_test, y_test, args.seed)

    setup_mlflow(args.tracking_uri, args.experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"actividad": "03-codigo-experimentacion", "tipo_problema": "regresion",
                         "autor": "Diego Falcon", "dataset": "kaggle_diamonds"})

        # Parámetros: hiperparámetros del modelo y configuración de datos
        for nombre, valor in {**model_params(args), "seed": args.seed, "test_size": args.test_size,
                              "n_train": len(X_train), "n_test": len(X_test),
                              "features": ",".join(FEATURES_NUM + FEATURES_CAT)}.items():
            mlflow.log_param(nombre, valor)

        # Métricas de evaluación (el tiempo se registra aparte: es lo único que no es determinista)
        for nombre, valor in metricas.items():
            mlflow.log_metric(nombre, valor)
        mlflow.log_metric("tiempo_entrenamiento_s", tiempo_entrenamiento)

        # Artefactos: archivo de métricas, gráficas y bitácora de limpieza
        resumen = {"run_name": run_name, "params": model_params(args), "seed": args.seed,
                   "metrics": metricas, "permutation_importance_rmse": importancias.round(2).to_dict()}
        artefactos = Path(tempfile.mkdtemp(prefix="tc5061_"))
        (artefactos / "metrics.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
        (artefactos / "bitacora_limpieza.txt").write_text(
            "\n".join(f"{paso}: {n:,} registros" for paso, n in bitacora), encoding="utf-8")
        fig_diag.savefig(artefactos / "diagnostico_residuos.png", dpi=110)
        fig_imp.savefig(artefactos / "importancia_permutacion.png", dpi=110)
        plt.close("all")

        mlflow.log_artifact(str(artefactos / "metrics.json"))
        mlflow.log_artifact(str(artefactos / "bitacora_limpieza.txt"), artifact_path="limpieza")
        mlflow.log_artifact(str(artefactos / "diagnostico_residuos.png"), artifact_path="graficas")
        mlflow.log_artifact(str(artefactos / "importancia_permutacion.png"), artifact_path="graficas")

        # Modelo: pipeline completo con firma; las categóricas se envían como texto.
        # skops (formato seguro por defecto en MLflow 3) exige declarar los tipos confiables; el modelo
        # lo genera este mismo script, así que se confía en los nodos de los árboles y en np.log/np.exp.
        X_firma = X_train.astype({c: str for c in FEATURES_CAT})
        mlflow.sklearn.log_model(pipeline, name="modelo",
                                 signature=infer_signature(X_firma, pred_train),
                                 input_example=X_firma.head(3),
                                 skops_trusted_types=SKOPS_TRUSTED_TYPES)

    print(f"Corrida '{run_name}' registrada en el experimento '{args.experiment_name}' (run_id {run.info.run_id})")
    print("  " + " | ".join(f"{k}={metricas[k]:,.3f}" for k in ["test_rmse", "test_mae", "test_r2"]))
    return run.info.run_id, metricas


if __name__ == "__main__":
    main()
