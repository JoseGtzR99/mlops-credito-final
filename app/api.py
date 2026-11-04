"""Servicio de inferencia: carga el modelo desde el Model Registry de MLflow
por ALIAS (models:/<nombre>@<alias>), nunca desde una ruta de archivo."""
from __future__ import annotations

import os
import socket
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from pydantic import BaseModel, Field

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "credito-riesgo")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")
POD_NAME = os.getenv("POD_NAME", socket.gethostname())  # en K8s se inyecta con downward API

FEATURES = ["ingreso_mensual", "deuda_actual", "antiguedad_laboral", "monto_solicitado", "plazo_meses"]

estado: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    mlflow.set_tracking_uri(TRACKING_URI)
    uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    estado["modelo"] = mlflow.sklearn.load_model(uri)
    estado["info"] = {
        "modelo": MODEL_NAME,
        "alias": MODEL_ALIAS,
        "version_registry": mv.version,
        "run_id": mv.run_id,
    }
    yield


app = FastAPI(title="API de riesgo crediticio", version="2.0.0", lifespan=lifespan)


class Solicitante(BaseModel):
    ingreso_mensual: float = Field(gt=0, description="Ingreso mensual (USD)")
    deuda_actual: float = Field(ge=0, description="Deuda actual (USD)")
    antiguedad_laboral: float = Field(ge=0, description="Antigüedad laboral (años)")
    monto_solicitado: float = Field(gt=0, description="Monto solicitado (USD)")
    plazo_meses: int = Field(gt=0, description="Plazo en meses")


@app.get("/health")
def health():
    if "modelo" not in estado:
        raise HTTPException(status_code=503, detail="modelo no cargado")
    return {"status": "ok", "pod": POD_NAME}


@app.get("/modelo")
def modelo_info():
    return {**estado.get("info", {}), "pod": POD_NAME}


@app.post("/predecir")
def predecir(s: Solicitante):
    X = pd.DataFrame([s.model_dump()], columns=FEATURES).astype(float)
    proba = float(estado["modelo"].predict_proba(X)[0][1])
    return {
        "moroso": int(proba >= 0.5),
        "probabilidad": proba,
        "version_modelo": estado["info"]["version_registry"],
        "run_id": estado["info"]["run_id"],
        "pod": POD_NAME,
    }
