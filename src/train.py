"""Entrenamiento con trazabilidad completa en MLflow.

- 6 ejecuciones con distintos hiperparámetros (RandomForest y GradientBoosting).
- Todas se evalúan sobre el MISMO split train/test (semilla fija, 70/30 estratificado).
- El run baseline y el mejor run se registran en el Model Registry
  ("credito-riesgo"), generando más de una versión; el mejor recibe el alias
  `production`, que es la referencia que consume el servicio.

Uso:  python -m src.train
"""
from __future__ import annotations

import os

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.data import FEATURES, TARGET, generar_datos

SEED = 42
N = 3000
TEST_SIZE = 0.30
EXPERIMENT = "credito-riesgo"
MODEL_NAME = "credito-riesgo"
ALIAS = "production"
METRICA_PRINCIPAL = "roc_auc"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

# (nombre del run, clase, hiperparámetros). El primero actúa como baseline.
CONFIGS = [
    ("rf_baseline", RandomForestClassifier, dict(n_estimators=20, max_depth=3)),
    ("rf_100_d6", RandomForestClassifier, dict(n_estimators=100, max_depth=6)),
    ("rf_150_d10", RandomForestClassifier, dict(n_estimators=150, max_depth=10)),
    ("gb_100_lr010_d3", GradientBoostingClassifier, dict(n_estimators=100, learning_rate=0.10, max_depth=3)),
    ("gb_200_lr005_d3", GradientBoostingClassifier, dict(n_estimators=200, learning_rate=0.05, max_depth=3)),
    ("gb_300_lr003_d4", GradientBoostingClassifier, dict(n_estimators=300, learning_rate=0.03, max_depth=4)),
]


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()

    df = generar_datos(N, seed=SEED)
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=TEST_SIZE, random_state=SEED, stratify=df[TARGET]
    )
    # Baseline de referencia para las pruebas de data drift
    os.makedirs("data", exist_ok=True)
    X_train.assign(**{TARGET: y_train}).to_csv("data/baseline_train.csv", index=False)

    resultados = []
    for nombre, clase, params in CONFIGS:
        with mlflow.start_run(run_name=nombre) as run:
            modelo = clase(random_state=SEED, **params).fit(X_train, y_train)
            proba = modelo.predict_proba(X_test)[:, 1]
            pred = modelo.predict(X_test)
            metricas = {
                "accuracy": accuracy_score(y_test, pred),
                "f1": f1_score(y_test, pred),
                "roc_auc": roc_auc_score(y_test, proba),
            }
            mlflow.log_params({"algoritmo": clase.__name__, **params})
            mlflow.log_params({"seed": SEED, "n_muestras": N, "test_size": TEST_SIZE})
            mlflow.log_metrics(metricas)
            mlflow.log_artifact("data/baseline_train.csv", artifact_path="datos")
            mlflow.sklearn.log_model(
                modelo,
                name="model",
                serialization_format="cloudpickle",
                signature=infer_signature(X_train, modelo.predict(X_train)),
                input_example=X_train.head(3),
            )
            resultados.append((nombre, run.info.run_id, metricas))
            print(f"[{nombre}] " + "  ".join(f"{k}={v:.4f}" for k, v in metricas.items()))

    mejor = max(resultados, key=lambda r: r[2][METRICA_PRINCIPAL])
    print(f"\nMejor run por {METRICA_PRINCIPAL}: {mejor[0]} ({mejor[2][METRICA_PRINCIPAL]:.4f})")

    # Registro: baseline (versión 1 en un registry nuevo) + mejor run
    a_registrar = [] if mejor[1] == resultados[0][1] else [resultados[0]]
    a_registrar.append(mejor)
    versiones = {}
    for nombre, run_id, _ in a_registrar:
        mv = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
        client.set_model_version_tag(MODEL_NAME, mv.version, "run_name", nombre)
        versiones[run_id] = mv.version
        print(f"Registrado {nombre} -> {MODEL_NAME} v{mv.version}")

    client.set_registered_model_alias(MODEL_NAME, ALIAS, versiones[mejor[1]])
    print(f"Alias '{ALIAS}' -> versión {versiones[mejor[1]]} (run {mejor[1]})")


if __name__ == "__main__":
    main()
