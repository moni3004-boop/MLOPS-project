FROM python:3.11-slim

WORKDIR /app

# System deps (joblib/scikit-learn sometimes needs these basics)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir uvicorn[standard] prometheus_client

# Copy source + artifacts
COPY src /app/src
COPY model_out /app/model_out
COPY artifacts /app/artifacts

# FastAPI listens on 8080
EXPOSE 8080

# Start API
CMD ["uvicorn", "src.serve_api:app", "--host", "0.0.0.0", "--port", "8080"]
