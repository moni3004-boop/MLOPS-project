FROM python:3.11-slim

WORKDIR /app
COPY . /app

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install python deps (WITHOUT torch here)
RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir pandas numpy scikit-learn scipy joblib fastapi "uvicorn[standard]" prometheus-client

# Install CPU-only torch explicitly (prevents CUDA downloads)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

EXPOSE 8080
CMD ["uvicorn", "src.serve_api:app", "--host", "0.0.0.0", "--port", "8080"]
