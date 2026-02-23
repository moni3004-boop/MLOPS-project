from pathlib import Path
import time

import joblib
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# ---- Paths (BEST artifacts) ----
MODEL_PATH = Path("model_out/best_model.pt")
PREP_PATH = Path("artifacts/preprocessor.joblib")

# ---- Load artifacts ----
bundle = torch.load(MODEL_PATH, map_location="cpu")
in_features = int(bundle["in_features"])
hidden = int(bundle["hidden"])
dropout = float(bundle["dropout"])
best_threshold = float(bundle.get("best_threshold", 0.5))

preprocessor = joblib.load(PREP_PATH)

# ---- Model definition must match training ----
class MLP(nn.Module):
    def __init__(self, in_features: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

model = MLP(in_features=in_features, hidden=hidden, dropout=dropout)
model.load_state_dict(bundle["state_dict"])
model.eval()

# ---- Prometheus metrics ----
REQ_COUNT = Counter("inference_requests_total", "Total inference requests")
REQ_LAT = Histogram("inference_latency_seconds", "Inference latency (seconds)")

app = FastAPI(title="Adult Income Inference API", version="1.0")

class PredictRequest(BaseModel):
    record: dict  # one row as dict

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
def predict(req: PredictRequest):
    REQ_COUNT.inc()
    start = time.time()

    import pandas as pd
    X = pd.DataFrame([req.record])

    Xt = preprocessor.transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    Xt = np.asarray(Xt, dtype=np.float32)

    with torch.no_grad():
        logits = model(torch.from_numpy(Xt))
        prob = torch.sigmoid(logits).item()
        pred = int(prob >= best_threshold)

    REQ_LAT.observe(time.time() - start)
    return {
        "probability_gt_50k": prob,
        "prediction": pred,
        "threshold": best_threshold
    }
