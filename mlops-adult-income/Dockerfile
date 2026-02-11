FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir pandas numpy scikit-learn scipy joblib torch fastapi uvicorn[standard] prometheus-client

EXPOSE 8080
CMD ["uvicorn", "src.serve_api:app", "--host", "0.0.0.0", "--port", "8080"]
