# Levanta el servidor MLflow (tracking + registry + proxy de artefactos) en el host.
# Los pods de Kubernetes lo consumen vía http://host.docker.internal:5000
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$env:PYTHONUTF8 = "1"
$env:MLFLOW_DISABLE_AGENT_HINT = "1"
$env:MLFLOW_SERVER_ALLOWED_HOSTS = "*"   # acepta Host: host.docker.internal

& .\.venv\Scripts\python.exe -m mlflow server `
    --host 0.0.0.0 --port 5000 `
    --backend-store-uri sqlite:///mlflow.db `
    --serve-artifacts --artifacts-destination ./mlartifacts
