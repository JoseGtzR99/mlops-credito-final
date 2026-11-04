"""Generador del dataset sintético de riesgo crediticio y de escenarios de deriva.

Es una versión parametrizable del `entrenar.py` del proyecto base: la regla
generadora de la etiqueta (concept) y las distribuciones de entrada (data) se
pueden alterar de forma controlada para simular drift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "ingreso_mensual",
    "deuda_actual",
    "antiguedad_laboral",
    "monto_solicitado",
    "plazo_meses",
]
TARGET = "moroso"
PLAZOS = [6, 12, 18, 24, 36, 48, 60]


def generar_datos(
    n: int,
    seed: int,
    data_drift: bool = False,
    concept_drift: bool = False,
) -> pd.DataFrame:
    """Genera `n` solicitudes de crédito.

    data_drift:    cambia la distribución de las ENTRADAS (ingresos menores,
                   deuda y montos mayores, plazos más largos).
    concept_drift: cambia la REGLA que genera la etiqueta, P(y|X), sin tocar
                   las entradas: la antigüedad deja de proteger y la carga
                   financiera pesa casi el doble (cambio de política/recesión).
    """
    rng = np.random.default_rng(seed)

    if data_drift:
        ingreso = rng.gamma(shape=3.0, scale=330.0, size=n) + 250  # ingresos ~35% menores
        deuda = rng.gamma(shape=2.6, scale=1100.0, size=n)  # más deuda
        monto = rng.gamma(shape=2.9, scale=2700.0, size=n) + 500  # montos mayores
        plazo = rng.choice(PLAZOS, size=n, p=[0.02, 0.05, 0.08, 0.15, 0.25, 0.25, 0.20])
    else:
        ingreso = rng.gamma(shape=4.0, scale=400.0, size=n) + 300
        deuda = rng.gamma(shape=2.0, scale=900.0, size=n)
        monto = rng.gamma(shape=2.5, scale=2200.0, size=n) + 500
        plazo = rng.choice(PLAZOS, size=n)
    antig = rng.exponential(scale=5.0, size=n).clip(0, 35)

    cuota = monto / plazo
    carga = (deuda * 0.04 + cuota) / ingreso
    ruido = rng.normal(0, 0.35, n)

    if concept_drift:
        score = 3.8 * carga + 0.02 * antig + ruido
        y = (score > 0.75).astype(int)
    else:
        score = 2.2 * carga - 0.06 * antig + ruido
        y = (score > 0.55).astype(int)

    return pd.DataFrame(
        {
            "ingreso_mensual": ingreso,
            "deuda_actual": deuda,
            "antiguedad_laboral": antig,
            "monto_solicitado": monto,
            "plazo_meses": plazo.astype(float),
            TARGET: y,
        }
    )
