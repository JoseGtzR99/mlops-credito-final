"""Utilidades compartidas por las pruebas de drift."""
from __future__ import annotations

import os

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from src.data import FEATURES, TARGET, generar_datos
from src.train import N, SEED, TEST_SIZE

BASELINE_CSV = "data/baseline_train.csv"


def cargar_baseline() -> pd.DataFrame:
    """Conjunto de entrenamiento de referencia (el mismo que vio el modelo)."""
    if os.path.exists(BASELINE_CSV):
        return pd.read_csv(BASELINE_CSV)
    df = generar_datos(N, seed=SEED)
    X_tr, _, y_tr, _ = train_test_split(
        df[FEATURES], df[TARGET], test_size=TEST_SIZE, random_state=SEED, stratify=df[TARGET]
    )
    return X_tr.assign(**{TARGET: y_tr})


def modelo_produccion(usar_registry: bool = True):
    """Modelo `production` del registry de MLflow; si el servidor no está
    disponible (p. ej. en CI), reentrena localmente la misma configuración."""
    if usar_registry:
        try:
            import mlflow

            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
            return mlflow.sklearn.load_model("models:/credito-riesgo@production"), "registry"
        except Exception:
            pass
    base = cargar_baseline()
    m = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.10, max_depth=3, random_state=SEED
    ).fit(base[FEATURES], base[TARGET])
    return m, "local"
