"""UI de consumo: llama a la API REAL desplegada en Kubernetes (Service credito-service)."""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://credito-service")

st.set_page_config(page_title="Riesgo crediticio", page_icon="🏦")
st.title("🏦 Evaluación de riesgo crediticio")
st.caption(f"API: {API_URL}")

try:
    info = requests.get(f"{API_URL}/modelo", timeout=5).json()
    st.info(
        f"Modelo **{info['modelo']}** · versión del registry **v{info['version_registry']}** "
        f"(alias `{info['alias']}`) · run `{info['run_id'][:8]}`"
    )
except Exception as e:  # noqa: BLE001
    st.error(f"No se pudo contactar la API: {e}")

with st.form("solicitud"):
    c1, c2 = st.columns(2)
    ingreso = c1.number_input("Ingreso mensual (USD)", min_value=1.0, value=1800.0, step=100.0)
    deuda = c2.number_input("Deuda actual (USD)", min_value=0.0, value=3500.0, step=100.0)
    antig = c1.number_input("Antigüedad laboral (años)", min_value=0.0, value=4.0, step=0.5)
    monto = c2.number_input("Monto solicitado (USD)", min_value=1.0, value=9000.0, step=500.0)
    plazo = st.selectbox("Plazo (meses)", [6, 12, 18, 24, 36, 48, 60], index=3)
    enviar = st.form_submit_button("Evaluar")

if enviar:
    payload = {
        "ingreso_mensual": ingreso,
        "deuda_actual": deuda,
        "antiguedad_laboral": antig,
        "monto_solicitado": monto,
        "plazo_meses": plazo,
    }
    r = requests.post(f"{API_URL}/predecir", json=payload, timeout=10)
    if r.ok:
        d = r.json()
        if d["moroso"]:
            st.error(f"Riesgo ALTO de mora · probabilidad {d['probabilidad']:.1%}")
        else:
            st.success(f"Riesgo BAJO de mora · probabilidad {d['probabilidad']:.1%}")
        st.caption(f"Atendido por el pod `{d['pod']}` · modelo v{d['version_modelo']}")
    else:
        st.error(f"Error {r.status_code}: {r.text}")
