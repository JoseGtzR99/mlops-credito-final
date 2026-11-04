"""Detección de CONCEPT DRIFT: degradación de la métrica sobre lotes sucesivos.

Escenario: durante los primeros LOTES_ESTABLES lotes la regla generadora de la
etiqueta es la de entrenamiento; a partir de ahí cambia (la antigüedad deja de
proteger y la carga financiera pesa casi el doble) SIN que las entradas cambien.

Métrica principal: F1 de la clase morosa. El accuracy no sirve: con una clase
minoritaria, un modelo que falla en ella conserva un accuracy aparentemente
alto y oculta la degradación.

Criterio de reentrenamiento (alarma)
------------------------------------
Se dispara cuando el F1 cae más de CAIDA_RELATIVA (10%) por debajo del F1 de
referencia (el del concepto de entrenamiento) durante LOTES_SOSTENIDOS (3)
lotes CONSECUTIVOS. El F1 de un lote de 500 tiene desviación ~0.035 (~5%): una
caída del 10% (~2 sigma) puede ocurrir por azar en un lote (~2%), pero 3 seguidas
son improbables (~1e-5): se evita reentrenar por un lote atípico.

Retraso de etiquetas
--------------------
En producción la etiqueta llega con retraso (aquí RETRASO_ETIQUETAS lotes: la
mora se confirma meses después). La métrica del lote t solo puede calcularse en
t + retraso, así que la alarma se emite tarde por construcción.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.data import FEATURES, TARGET, generar_datos

N_LOTES = 14
LOTES_ESTABLES = 6  # lotes 0..5 sin cambio; el concepto cambia desde el lote 6
TAM_LOTE = 500
CAIDA_RELATIVA = 0.10
LOTES_SOSTENIDOS = 3
RETRASO_ETIQUETAS = 2


@dataclass
class ResultadoConcepto:
    metricas: pd.DataFrame
    f1_referencia: float
    umbral_f1: float
    lote_alarma: int | None  # lote cuya métrica completa el criterio
    lote_alarma_con_retraso: int | None  # lote en que se emite realmente la alarma


def simular_lotes(
    modelo, f1_referencia: float, seed: int = 1000, con_deriva: bool = True
) -> ResultadoConcepto:
    filas = []
    for t in range(N_LOTES):
        deriva = con_deriva and t >= LOTES_ESTABLES
        lote = generar_datos(TAM_LOTE, seed=seed + t, concept_drift=deriva)
        X, y = lote[FEATURES], lote[TARGET]
        pred = modelo.predict(X)
        proba = modelo.predict_proba(X)[:, 1]
        filas.append(
            {
                "lote": t,
                "concepto_cambiado": deriva,
                "pct_morosos": y.mean(),
                "accuracy": accuracy_score(y, pred),
                "f1": f1_score(y, pred),
                "roc_auc": roc_auc_score(y, proba),
                "prob_media_predicha": proba.mean(),
            }
        )
    m = pd.DataFrame(filas)
    umbral = f1_referencia * (1 - CAIDA_RELATIVA)
    bajo = (m["f1"] < umbral).astype(int)
    racha = bajo.groupby((bajo == 0).cumsum()).cumsum()
    idx = np.flatnonzero(racha.to_numpy() >= LOTES_SOSTENIDOS)
    alarma = int(idx[0]) if len(idx) else None
    return ResultadoConcepto(
        m, f1_referencia, umbral, alarma, None if alarma is None else alarma + RETRASO_ETIQUETAS
    )


def f1_referencia(modelo, seed: int = 999, n: int = 2000) -> float:
    """F1 de referencia: lote grande del MISMO concepto de entrenamiento."""
    d = generar_datos(n, seed=seed)
    return float(f1_score(d[TARGET], modelo.predict(d[FEATURES])))
