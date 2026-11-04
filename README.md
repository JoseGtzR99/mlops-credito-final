# MLOps — Riesgo crediticio: MLflow + Docker + Kubernetes + Drift

Proyecto final de ML Ops (trabajo individual). Lleva un modelo de riesgo
crediticio (clasificación tabular, 5 variables, 3 000 registros sintéticos) desde
el entrenamiento hasta un despliegue con 3 réplicas en Kubernetes y monitoreo de
drift. Detalle de decisiones en [`docs/ARQUITECTURA.md`](docs/ARQUITECTURA.md).

```
 src/train.py ──► MLflow (tracking + registry, alias "production")
                         │  models:/credito-riesgo@production
                         ▼
 Dockerfile ──► imagen credito-api ──► Deployment x3 ──► Service ──► UI Streamlit
                                       (K8s, Docker Desktop)
 drift/ ──► data drift (KS / chi2 / PSI) · concept drift (F1 por lotes)
```

## Estructura

| Ruta | Contenido |
|---|---|
| `src/data.py` | Generador del dataset y de los escenarios de deriva |
| `src/train.py` | 6 runs en MLflow, registro de versiones y alias `production` |
| `app/api.py` | API FastAPI; carga el modelo por alias; devuelve el pod que responde |
| `Dockerfile`, `requirements.txt` | Imagen de la API, dependencias fijadas (incluye `mlflow==3.18.0`) |
| `k8s/` | `deployment.yaml` (3 réplicas, probes), `service.yaml`, `ui.yaml` |
| `drift/` | Pruebas de data drift y concept drift + generador de evidencia |
| `tests/test_drift.py` | Pruebas automatizadas (verde/rojo) |
| `ui/` | UI Streamlit (extra) que consume el Service en Kubernetes |
| `scripts/` | Arranque de MLflow y generación de evidencias |
| `docs/` | Arquitectura y evidencias |

## Reproducir paso a paso (Windows / PowerShell)

```powershell
py -3.11 -m venv .venv ; .\.venv\Scripts\pip install -r requirements-dev.txt
$env:PYTHONUTF8 = 1

# 1. MLflow (dejar corriendo en otra terminal) -> UI en http://127.0.0.1:5000
powershell -ExecutionPolicy Bypass -File .\scripts\start_mlflow.ps1

# 2. Entrenar: 6 runs + registro de v1 (baseline) y v2 (mejor) con alias production
.\.venv\Scripts\python -m src.train

# 3. Imagen y despliegue en Kubernetes (Docker Desktop, contexto docker-desktop)
docker build -t credito-api:1.0.0 .
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml

# 4. Evidencias de Kubernetes (Running, balanceo, autorreparación, escalado)
powershell -ExecutionPolicy Bypass -File .\scripts\evidencias_k8s.ps1

# 5. Drift
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m drift.run_report

# 6. UI (extra), desplegada en el clúster
docker build -t credito-ui:1.0.0 ui
kubectl apply -f k8s/ui.yaml
kubectl port-forward svc/credito-ui 8501:8501      # http://localhost:8501
```

Los pods acceden a MLflow en el host mediante `http://host.docker.internal:5000`;
por eso el servidor MLflow debe estar corriendo antes de crear/recrear pods.

## Probar la API dentro del clúster

```powershell
kubectl run t --rm -i --restart=Never --image=curlimages/curl:8.10.1 -- curl -s http://credito-service/modelo
```
