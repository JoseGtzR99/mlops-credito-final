FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

# El modelo NO va en la imagen: se descarga del Model Registry (alias) al arrancar.
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
