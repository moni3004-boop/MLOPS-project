FROM python:3.11-slim

# Prevent Python from writing .pyc files & enable stdout logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy only requirements first (better layer caching)
COPY requirements-api.txt .

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install app dependencies (without torch here)
RUN pip install --no-cache-dir -r requirements-api.txt

# Install CPU-only torch explicitly (prevents CUDA downloads)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# Now copy the rest of the application
COPY . .

EXPOSE 8080

CMD ["uvicorn", "src.serve_api:app", "--host", "0.0.0.0", "--port", "8080"]
