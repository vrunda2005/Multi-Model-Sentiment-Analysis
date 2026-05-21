
"""
FastAPI REST backend for the Sentiment Analysis project.
Endpoints:
  GET  /health          — server status
  GET  /models          — list available models
  POST /predict         — single text prediction
  POST /predict/batch   — batch predictions
  POST /predict/compare — all models compared
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.predictor import get_predictor
from src.utils.helpers import get_config, get_logger, device_info

logger = get_logger("api")
cfg    = get_config()

# ──────────────────────────────────────────────────────────
#  App Setup
# ──────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Sentiment Analysis API",
    description = "Multi-tier sentiment analysis: TF-IDF Baseline | BiLSTM | DistilBERT",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────
#  Request / Response Schemas
# ──────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000,
                      example="This movie was absolutely fantastic!")
    model: Optional[str] = Field(None, example="transformer",
                                 description="baseline | lstm | transformer")

class PredictBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_items=100,
                             example=["Great product!", "Terrible experience."])
    model: Optional[str] = Field(None)

class PredictResponse(BaseModel):
    label:         str
    confidence:    float
    probabilities: Dict[str, float]
    latency_ms:    float
    model_used:    str

class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]
    total_latency_ms: float
    n_samples: int

class CompareResponse(BaseModel):
    text: str
    results: Dict[str, PredictResponse]

class HealthResponse(BaseModel):
    status: str
    available_models: List[str]
    device: str


# ──────────────────────────────────────────────────────────
#  Routes
# ──────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Status"])
def health():
    """Server health and available models."""
    predictor = get_predictor()
    info      = device_info()
    return HealthResponse(
        status           = "ok",
        available_models = predictor.available_models,
        device           = info["device"],
    )


@app.get("/models", tags=["Status"])
def list_models():
    """Return list of available trained models."""
    predictor = get_predictor()
    return {"available_models": predictor.available_models}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(req: PredictRequest):
    """
    Predict sentiment for a single text.
    - **text**: Input text (1–10,000 characters)
    - **model**: Model tier to use (defaults to config default)
    """
    predictor = get_predictor()
    try:
        result = predictor.predict(req.text, req.model)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result.to_dict()


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
def predict_batch(req: PredictBatchRequest):
    """
    Predict sentiment for up to 100 texts at once.
    """
    import time
    predictor = get_predictor()

    predictions = []
    t0 = time.time()
    for text in req.texts:
        try:
            r = predictor.predict(text, req.model)
            predictions.append(r.to_dict())
        except Exception as e:
            predictions.append({
                "label": "error", "confidence": 0.0,
                "probabilities": {}, "latency_ms": 0.0,
                "model_used": str(req.model), "error": str(e)
            })
    total_ms = (time.time() - t0) * 1000

    return BatchPredictResponse(
        predictions      = predictions,
        total_latency_ms = round(total_ms, 2),
        n_samples        = len(predictions),
    )


@app.post("/predict/compare", response_model=CompareResponse, tags=["Prediction"])
def predict_compare(req: PredictRequest):
    """
    Run the same text through ALL available models and return comparison.
    """
    predictor = get_predictor()
    try:
        results_raw = predictor.predict_all(req.text)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return CompareResponse(
        text    = req.text,
        results = {name: res.to_dict() for name, res in results_raw.items()},
    )


# ──────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host   = cfg["api"]["host"],
        port   = cfg["api"]["port"],
        reload = True,
    )
