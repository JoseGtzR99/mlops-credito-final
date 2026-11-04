"""Genera la evidencia de drift: tablas, CSV y gráficas en docs/evidencias/.

Uso:  python -m drift.run_report
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drift.common import cargar_baseline, modelo_produccion
from drift.concept_drift import (
    CAIDA_RELATIVA,
    LOTES_ESTABLES,
    RETRASO_ETIQUETAS,
    f1_referencia,
    simular_lotes,
)
from drift.data_drift import evaluar_data_drift, tabla
from src.data import generar_datos

OUT = "docs/evidencias"


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    base = cargar_baseline()
    modelo, origen = modelo_produccion()
    print(f"Modelo cargado desde: {origen}")

    # ------------------------------ DATA DRIFT ------------------------------
    escenarios = {
        "mismo_origen": generar_datos(1000, seed=777),
        "con_deriva_inyectada": generar_datos(1000, seed=778, data_drift=True),
    }
    resumen = {}
    with open(f"{OUT}/data_drift.txt", "w", encoding="utf-8") as f:
        for nombre, lote in escenarios.items():
            estado, res = evaluar_data_drift(base, lote)
            resumen[nombre] = estado
            f.write(f"=== {nombre}: {estado} ===\n")
            f.write(tabla(res).to_string(index=False, float_format=lambda x: f"{x:.4g}") + "\n\n")
            print(f"data drift [{nombre}] -> {estado}")
    with open(f"{OUT}/data_drift_estado.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2)

    # ----------------------------- CONCEPT DRIFT ----------------------------
    f1_ref = f1_referencia(modelo)
    r = simular_lotes(modelo, f1_ref)
    r.metricas.to_csv(f"{OUT}/concept_drift_lotes.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(r.metricas["lote"], r.metricas["f1"], "o-", label="F1 (clase morosa)", color="#1f77b4")
    ax.plot(r.metricas["lote"], r.metricas["accuracy"], "s--", label="Accuracy", color="#7f7f7f", alpha=0.7)
    ax.axhline(r.umbral_f1, color="#d62728", ls=":", label=f"Umbral F1 = {r.umbral_f1:.2f} (-{CAIDA_RELATIVA:.0%})")
    ax.axvline(LOTES_ESTABLES - 0.5, color="#ff7f0e", ls="--", label="Cambio de concepto")
    if r.lote_alarma is not None:
        ax.axvline(r.lote_alarma, color="#d62728", lw=2, alpha=0.5,
                   label=f"Criterio cumplido (3 lotes sostenidos): lote {r.lote_alarma}")
        ax.axvline(r.lote_alarma_con_retraso, color="#9467bd", lw=2, alpha=0.5,
                   label=f"Alarma real con retraso de etiquetas ({RETRASO_ETIQUETAS}): lote {r.lote_alarma_con_retraso}")
    ax.set_xlabel("Lote de producción")
    ax.set_ylabel("Métrica")
    ax.set_title("Concept drift: degradación de la métrica por lote")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/concept_drift.png", dpi=150)

    print(r.metricas.round(3).to_string(index=False))
    print(f"F1 referencia={f1_ref:.3f} umbral={r.umbral_f1:.3f} "
          f"alarma_lote={r.lote_alarma} alarma_con_retraso={r.lote_alarma_con_retraso}")


if __name__ == "__main__":
    main()
