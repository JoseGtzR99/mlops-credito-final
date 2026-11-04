# Documento de arquitectura

## 1. Resumen y reparto de trabajo

Proyecto **individual**. Autor: JoseGtzR99. Todos los componentes fueron
construidos por el autor; el historial de commits lo refleja.

| Componente | Responsable |
|---|---|
| Fase 1 — MLflow (entrenamiento, registry) | JoseGtzR99 |
| Fase 2 — Docker | JoseGtzR99 |
| Fase 3 — Kubernetes | JoseGtzR99 |
| Drift (data + concept) y pruebas | JoseGtzR99 |
| Extra — UI | JoseGtzR99 |

## 2. Problema y datos

Clasificación binaria de morosidad (`moroso` 0/1) con 5 variables numéricas:
`ingreso_mensual`, `deuda_actual`, `antiguedad_laboral`, `monto_solicitado`, `plazo_meses`.
Dataset sintético (reutiliza la regla generadora del proyecto base del curso, ahora
parametrizable en `src/data.py`) de 3 000 registros, ~28 % de morosos. Se eligió sintético
porque permite inyectar data drift y concept drift de forma controlada y reproducible.

Split: 70 / 30 estratificado, `random_state=42`. El conjunto de entrenamiento (2 100 filas)
es el baseline de referencia para data drift (`data/baseline_train.csv`, también adjunto
como artefacto a cada run).

## 3. Fase 1 — MLflow

* Servidor local: SQLite como backend + proxy de artefactos (`--serve-artifacts`), de modo
  que clientes remotos (los pods) descargan el modelo por HTTP sin compartir disco.
* Experimento `credito-riesgo`, **6 runs** (misma semilla y mismo split → comparables):

| Run | Algoritmo | Hiperparámetros | Accuracy | F1 | ROC AUC |
|---|---|---|---|---|---|
| rf_baseline | RandomForest | n=20, depth=3 | 0.8044 | 0.5533 | 0.8767 |
| rf_100_d6 | RandomForest | n=100, depth=6 | 0.8578 | 0.7064 | 0.9021 |
| rf_150_d10 | RandomForest | n=150, depth=10 | 0.8578 | 0.7253 | 0.8950 |
| **gb_100_lr010_d3** | GradientBoosting | n=100, lr=0.10, depth=3 | 0.8733 | 0.7543 | **0.9075** |
| gb_200_lr005_d3 | GradientBoosting | n=200, lr=0.05, depth=3 | 0.8700 | 0.7484 | 0.9068 |
| gb_300_lr003_d4 | GradientBoosting | n=300, lr=0.03, depth=4 | 0.8733 | 0.7574 | 0.9045 |

* **Métrica principal: ROC AUC** (independiente del umbral y robusta al desbalance 72/28).
  El run `gb_100_lr010_d3` gana por AUC; además tiene accuracy máxima (empatada) y un F1 a
  0.003 del mejor, con un modelo más simple que `gb_300` (100 vs 300 árboles, menos
  latencia y memoria). Es el criterio para elegirlo.
* Model Registry, nombre estable **`credito-riesgo`**:

| Versión | Run | Descripción |
|---|---|---|
| v1 | `rf_baseline` (`d962398704e2…`) | baseline |
| **v2** | **`gb_100_lr010_d3` (`da784e0147f34776a224e3204f2cf643`)** | **alias `production`** |

* **Versión desplegada: `credito-riesgo` v2 ← run `da784e0147f34776a224e3204f2cf643`.**
  El servicio carga `models:/credito-riesgo@production`; el endpoint `/modelo` y cada
  respuesta de `/predecir` devuelven `version_registry` y `run_id`, de modo que cualquier
  predicción es rastreable hasta el experimento que la produjo.
* Promover o revertir un modelo = mover el alias (`set_registered_model_alias`) y reiniciar
  los pods (`kubectl rollout restart deploy/credito-api`); no se reconstruye la imagen.
* Los modelos se serializan con `cloudpickle` (formato de MLflow sklearn); `skops` rechaza
  por defecto el tipo `sklearn.tree._tree.Tree`. Es aceptable porque el registry es propio
  y de confianza.

## 4. Fase 2 — Docker

`Dockerfile` propio sobre `python:3.11-slim`; `requirements.txt` con todas las versiones
fijadas (`mlflow==3.18.0`, `scikit-learn==1.9.1`, …). La versión de scikit-learn es la misma
del entrenamiento: evita incompatibilidades al deserializar. **El modelo no se hornea en la
imagen**: se descarga del registry al arrancar (`lifespan` de FastAPI), por lo que la imagen
es genérica y la versión servida queda gobernada por el alias. El contenedor arranca solo
con `uvicorn`, sin pasos manuales. Falla de forma explícita (readiness en rojo) si no puede
alcanzar MLflow.

## 5. Fase 3 — Kubernetes (Docker Desktop)

* `Deployment credito-api`: 3 réplicas, `requests/limits` de CPU y memoria, `readinessProbe`
  y `livenessProbe` en `/health`. `POD_NAME` se inyecta con *downward API* y se devuelve en
  cada respuesta.
* `Service credito-service` (NodePort 30090 → 8000): balancea entre los pods *Ready*.
* Los pods alcanzan MLflow (que corre en el host) vía `host.docker.internal:5000`.
* Evidencia en `docs/evidencias/` (generada por `scripts/evidencias_k8s.ps1`, las peticiones
  se lanzan **desde un pod dentro del clúster** contra el Service; `port-forward` fijaría un
  único pod y no demostraría balanceo):

| Demostración | Archivo | Resultado |
|---|---|---|
| ≥3 réplicas Running | `k8s_1_replicas_running.txt` | 3/3 Running |
| Balanceo | `k8s_2_balanceo.txt` | 30 peticiones repartidas 12/12/6 entre 3 pods |
| Autorreparación | `k8s_3_autorreparacion.txt` | Se elimina un pod; 20 peticiones simultáneas, 0 fallos; el pod se recrea |
| Escalado | `k8s_4_escalado.txt` | 3 → 5 réplicas: 5 pods atienden tráfico; luego 5 → 3 |

## 6. Drift

Código en `drift/`, pruebas en `tests/test_drift.py` (11 pruebas, todas en verde). Para que
corran sin infraestructura, las pruebas reentrenan localmente la misma configuración del
modelo de producción; `python -m drift.run_report` usa el modelo real del registry.

### 6.1 Data drift

| Variable | Prueba | Por qué |
|---|---|---|
| ingreso, deuda, antigüedad, monto (continuas, asimétricas) | Kolmogorov-Smirnov 2 muestras | No paramétrica: no supone normalidad; sensible a cambios de localización y de forma |
| plazo_meses (discreta, 7 niveles) | Chi-cuadrada de homogeneidad | KS no es apropiada con empates masivos; compara frecuencias por nivel |

**Umbral:** p-valor < 0.01 / 5 = **0.002** (alpha 0.01 con corrección de Bonferroni por las 5
pruebas simultáneas; sin ella la tasa de falsa alarma por lote sería ~5 % en lugar de ~1 %).
Se reporta además **PSI** con los cortes estándar de la banca (<0.10 estable, 0.10–0.25
moderado, ≥0.25 significativo) para dimensionar la magnitud, porque con muestras grandes un
p-valor pequeño puede corresponder a un cambio irrelevante.

Resultado (`docs/evidencias/data_drift.txt`): lote del mismo origen → **VERDE** (p-valores
0.08–0.69, PSI < 0.025); lote con deriva inyectada (ingresos ~35 % menores, más deuda y
monto, plazos más largos) → **ROJO** (4 de 5 variables; `antiguedad_laboral`, no alterada,
correctamente sin alerta).

### 6.2 Concept drift

Escenario: 14 lotes de 500. Lotes 0–5 con la regla de entrenamiento; desde el lote 6 la
regla cambia (la antigüedad deja de proteger, la carga financiera pesa ~1.7× más y el umbral
de mora baja) **sin cambiar las entradas** (verificado: el test de data drift sigue en
verde). Gráfica: `docs/evidencias/concept_drift.png`; datos: `concept_drift_lotes.csv`.

* **Métrica:** F1 de la clase morosa (el accuracy se degrada menos y oculta el problema en la
  clase minoritaria). F1 de referencia = 0.69; en lotes estables oscila 0.67–0.76.
* **Criterio de reentrenamiento:** F1 más de **10 %** por debajo de la referencia (F1 < 0.62)
  durante **3 lotes consecutivos**. El F1 de un lote de 500 tiene desviación ~0.035 (~5 %):
  una caída puntual del 10 % es posible por azar (~2 %), tres seguidas no (~1e-5).
* **Resultado:** el criterio se cumple en el lote 8 (dos lotes después del cambio), sin falsas
  alarmas en los lotes 0–5. Con un retraso de etiquetas de 2 lotes la alarma real se emite en
  el lote 10.

### 6.3 Retraso de etiquetas

En crédito la etiqueta (mora) se conoce meses después, así que la métrica anterior llega
tarde. Mientras tanto, el equipo:

1. **Vigila data drift** (no necesita etiquetas): detecta cambios de población, aunque no
   los de concepto puro. Hay que ser explícito: en este escenario el data drift está en
   verde mientras el modelo se degrada, por eso no basta.
2. **Monitorea la distribución de las predicciones** (probabilidad media, % de rechazos)
   y la tasa de aprobación; un cambio abrupto es señal temprana. Con concept drift puro
   tampoco se mueve (`prob_media_predicha` es estable en el CSV), así que es complementario.
3. **Usa etiquetas proxy tempranas**: mora a 30 días, no pago de la primera cuota, en vez de
   esperar la mora a 90 días; permiten calcular una métrica aproximada antes.
4. **Etiqueta una muestra** (revisión manual / auditoría) de forma sistemática.
5. **Evalúa con ventanas retrasadas** (la métrica del lote t se calcula en t+retraso) y asume
   el costo: durante ese lapso se opera con riesgo; por eso conviene un modelo *challenger*
   en sombra y planes de contingencia (reglas de negocio más conservadoras, límites de
   exposición) hasta confirmar.

## 7. Extra — UI

`ui/streamlit_app.py` (Streamlit) desplegada **como pod en el mismo clúster**
(`k8s/ui.yaml`); consume `http://credito-service` (el Service de Kubernetes, no un proceso
local) y muestra la versión del registry y el pod que atendió cada petición.
Acceso: `kubectl port-forward svc/credito-ui 8501:8501` → http://localhost:8501.

## 8. Limitaciones conocidas

* El servidor MLflow corre en el host; en producción real sería un servicio propio
  (con PostgreSQL y almacenamiento de objetos) y no un punto único de falla.
  Una vez cargado el modelo el pod no depende de MLflow, pero un pod nuevo sí.
* El NodePort 30090 no es alcanzable desde `localhost` en este clúster de Docker Desktop;
  las pruebas se hacen desde dentro del clúster o con `port-forward`.
* Datos sintéticos: las métricas absolutas no son comparables con un caso real.
