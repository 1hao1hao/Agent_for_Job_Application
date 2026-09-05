FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    EVALRAG_PROJECT_ROOT=/app

WORKDIR /app

COPY requirements-service.txt ./
RUN pip install --no-cache-dir -r requirements-service.txt

COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY migrations ./migrations
COPY data/evaluation ./data/evaluation
COPY data/processed/chunks ./data/processed/chunks

RUN mkdir -p reports/runs traces/service data/processed/indexes \
    && python scripts/build_bm25_index.py \
        --dataset-version evalrag_v0.2 \
        --output data/processed/indexes/evalrag_v0.2/bm25-v1/index.json \
    && python scripts/build_bm25_index.py \
        --dataset-version evalrag_v0.3 \
        --output data/processed/indexes/evalrag_v0.3/bm25-v1/index.json

EXPOSE 8000

CMD ["uvicorn", "intern_rag.serving.runtime:create_runtime_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
