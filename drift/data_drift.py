"""Detección de DATA DRIFT: lote nuevo vs. baseline de entrenamiento.

Pruebas por tipo de variable
----------------------------
* Continuas (ingreso, deuda, antigüedad, monto): Kolmogorov-Smirnov de dos
  muestras. Es no paramétrica (las variables son asimétricas: gamma/exponencial,
  no normales) y sensible a cambios de localización y de forma.
* Discreta/categórica (plazo_meses, 7 niveles): chi-cuadrada de homogeneidad
  sobre las frecuencias por nivel. KS no es apropiada con empates masivos.

Umbral de alerta
----------------
* p-valor < ALPHA / n_variables  (ALPHA = 0.01, corrección de Bonferroni).
  Con 5 pruebas simultáneas, un alpha ingenuo de 0.01 por variable elevaría la
  tasa de falsas alarmas por lote a ~5%; Bonferroni la mantiene en ~1%.
* PSI (Population Stability Index) se reporta como medida de MAGNITUD, con los
  cortes estándar de la industria bancaria: <0.10 estable, 0.10-0.25 moderado,
  >=0.25 cambio significativo. Un p-valor chico con n grande puede ser
  estadísticamente significativo pero irrelevante; el PSI acota eso.
El lote está en ROJO si alguna variable supera el umbral de p-valor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

CONTINUAS = ["ingreso_mensual", "deuda_actual", "antiguedad_laboral", "monto_solicitado"]
CATEGORICAS = ["plazo_meses"]
ALPHA = 0.01
PSI_MODERADO = 0.10
PSI_ALTO = 0.25


@dataclass
class ResultadoVariable:
    variable: str
    prueba: str
    estadistico: float
    p_valor: float
    psi: float
    deriva: bool


def psi(ref: pd.Series, nuevo: pd.Series, bins: int = 10) -> float:
    """PSI con cortes por deciles del baseline (continuas) o por nivel (discretas)."""
    if ref.nunique() <= 10:
        niveles = sorted(set(ref.unique()) | set(nuevo.unique()))
        p = np.array([(ref == v).mean() for v in niveles])
        q = np.array([(nuevo == v).mean() for v in niveles])
    else:
        cortes = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        cortes[0], cortes[-1] = -np.inf, np.inf
        p = np.histogram(ref, cortes)[0] / len(ref)
        q = np.histogram(nuevo, cortes)[0] / len(nuevo)
    p, q = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
    return float(np.sum((q - p) * np.log(q / p)))


def evaluar_data_drift(baseline: pd.DataFrame, lote: pd.DataFrame, alpha: float = ALPHA):
    """Devuelve (estado, resultados). estado = 'ROJO' | 'VERDE'."""
    variables = CONTINUAS + CATEGORICAS
    umbral = alpha / len(variables)
    resultados = []
    for v in variables:
        if v in CONTINUAS:
            r = stats.ks_2samp(baseline[v], lote[v])
            prueba, est, p = "KS", float(r.statistic), float(r.pvalue)
        else:
            niveles = sorted(set(baseline[v]) | set(lote[v]))
            tabla_f = np.array(
                [[(baseline[v] == n).sum() for n in niveles], [(lote[v] == n).sum() for n in niveles]]
            )
            chi2, p, *_ = stats.chi2_contingency(tabla_f)
            prueba, est, p = "chi2", float(chi2), float(p)
        resultados.append(
            ResultadoVariable(v, prueba, est, p, psi(baseline[v], lote[v]), bool(p < umbral))
        )
    estado = "ROJO" if any(r.deriva for r in resultados) else "VERDE"
    return estado, resultados


def tabla(resultados) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in resultados])
