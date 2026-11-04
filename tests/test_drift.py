"""Pruebas automatizadas de data drift y concept drift (no requieren MLflow)."""
import pytest

from drift.common import cargar_baseline, modelo_produccion
from drift.concept_drift import (
    LOTES_ESTABLES,
    LOTES_SOSTENIDOS,
    RETRASO_ETIQUETAS,
    f1_referencia,
    simular_lotes,
)
from drift.data_drift import evaluar_data_drift
from src.data import generar_datos


@pytest.fixture(scope="module")
def baseline():
    return cargar_baseline()


@pytest.fixture(scope="module")
def modelo():
    m, _ = modelo_produccion(usar_registry=False)
    return m


# ------------------------------- DATA DRIFT ---------------------------------
@pytest.mark.parametrize("seed", [11, 12, 13, 14, 15])
def test_data_drift_verde_con_datos_del_mismo_origen(baseline, seed):
    estado, res = evaluar_data_drift(baseline, generar_datos(1000, seed=seed))
    assert estado == "VERDE", [r for r in res if r.deriva]


@pytest.mark.parametrize("seed", [21, 22, 23])
def test_data_drift_rojo_con_datos_inyectados(baseline, seed):
    estado, res = evaluar_data_drift(baseline, generar_datos(1000, seed=seed, data_drift=True))
    assert estado == "ROJO"
    deriva = {r.variable for r in res if r.deriva}
    # La antigüedad NO se alteró en el escenario: no debe marcarse
    assert "antiguedad_laboral" not in deriva
    assert {"ingreso_mensual", "plazo_meses"} <= deriva


def test_concept_drift_no_altera_las_entradas(baseline):
    """Concept drift = cambia P(y|X), no P(X): el test de data drift debe seguir verde."""
    estado, _ = evaluar_data_drift(baseline, generar_datos(1000, seed=31, concept_drift=True))
    assert estado == "VERDE"


# ------------------------------ CONCEPT DRIFT -------------------------------
def test_concept_drift_sin_cambio_no_dispara_alarma(modelo):
    r = simular_lotes(modelo, f1_referencia(modelo), con_deriva=False)
    assert r.lote_alarma is None


def test_concept_drift_degrada_la_metrica_y_dispara_alarma(modelo):
    r = simular_lotes(modelo, f1_referencia(modelo))
    estables = r.metricas[r.metricas["lote"] < LOTES_ESTABLES]["f1"]
    derivados = r.metricas[r.metricas["lote"] >= LOTES_ESTABLES]["f1"]
    assert derivados.mean() < estables.mean() - 0.05  # degradación material
    assert (estables >= r.umbral_f1).all()  # sin falsas alarmas antes del cambio
    assert r.lote_alarma is not None and r.lote_alarma >= LOTES_ESTABLES
    # La alarma exige LOTES_SOSTENIDOS lotes consecutivos bajo el umbral
    assert r.lote_alarma >= LOTES_ESTABLES + LOTES_SOSTENIDOS - 1
    # Con retraso de etiquetas la alarma real llega más tarde
    assert r.lote_alarma_con_retraso == r.lote_alarma + RETRASO_ETIQUETAS
