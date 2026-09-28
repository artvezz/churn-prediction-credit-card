FROM python:3.12-slim

WORKDIR /app

COPY scoring-service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Servico + modulo de features/negocio (fonte unica com o treino)
COPY scoring-service/ /app/
COPY src/features.py src/business.py /app/

# Artefatos do treino (nao versionados; gerados por src/train.py)
COPY artifacts/churn_model.pkl /app/model/churn_model.pkl
COPY artifacts/feature_columns.json /app/model/feature_columns.json
COPY artifacts/medians.json /app/model/medians.json
COPY artifacts/business_params.json /app/model/business_params.json

ENV PYTHONPATH=/app \
    DATA_DIR=/data \
    CHURN_SOURCE=supabase

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]