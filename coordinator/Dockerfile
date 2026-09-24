FROM python:3.11-slim

WORKDIR /app

# System deps for sentence-transformers / chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Pre-warm & cache ChromaDB ONNX embedding model directly inside Docker image
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction; DefaultEmbeddingFunction()(['warmup'])"

# Copy source
COPY shared/ ./shared/
COPY coordinator/ ./coordinator/

EXPOSE 8000 9091

CMD ["python", "-m", "coordinator.app.main"]
