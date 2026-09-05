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
COPY data/processed/indexes/evalrag_v0.2/bm25-v1 ./data/processed/indexes/evalrag_v0.2/bm25-v1
COPY data/processed/indexes/evalrag_v0.3/bm25-v1 ./data/processed/indexes/evalrag_v0.3/bm25-v1

RUN mkdir -p reports/runs traces/service data/processed/indexes

EXPOSE 8000

CMD ["uvicorn", "intern_rag.serving.runtime:create_runtime_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
