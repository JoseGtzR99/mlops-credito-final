"""Vuelca a docs/evidencias/mlflow_*.txt los runs y las versiones del Model Registry."""
import os

import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
c = MlflowClient()
exp = c.get_experiment_by_name("credito-riesgo")
runs = mlflow.search_runs([exp.experiment_id], order_by=["metrics.roc_auc DESC"])
cols = ["run_id", "tags.mlflow.runName", "params.algoritmo", "params.n_estimators",
        "params.max_depth", "params.learning_rate", "metrics.accuracy", "metrics.f1", "metrics.roc_auc"]
os.makedirs("docs/evidencias", exist_ok=True)
with open("docs/evidencias/mlflow_runs.txt", "w", encoding="utf-8") as f:
    f.write(f"Experimento: {exp.name} (id {exp.experiment_id}) - ordenado por roc_auc desc\n\n")
    f.write(runs[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
with open("docs/evidencias/mlflow_registry.txt", "w", encoding="utf-8") as f:
    for mv in c.search_model_versions("name='credito-riesgo'"):
        f.write(f"credito-riesgo v{mv.version}  run_id={mv.run_id}  run_name={mv.tags.get('run_name')}\n")
    rm = c.get_registered_model("credito-riesgo")
    f.write(f"\nAliases: {dict(rm.aliases)}\n")
print(open("docs/evidencias/mlflow_runs.txt", encoding="utf-8").read())
print(open("docs/evidencias/mlflow_registry.txt", encoding="utf-8").read())
