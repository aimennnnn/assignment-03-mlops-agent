"""Prediction API (FastAPI) that serves the champion model on Kubernetes.

    GET  /health   liveness probe: is the process alive?
    GET  /ready    readiness probe: is the model loaded and safe to send traffic to?
    GET  /version  which app version and which model is running
    POST /predict  {"features": [30 numbers]} -> prediction
"""

import json
import logging
import os
import warnings
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Settings come from the Kubernetes ConfigMap (see kubernetes/configmap.yaml).
APP_VERSION = os.environ.get("APP_VERSION", "dev")
MODEL_PATH = os.environ.get("MODEL_PATH", "model/model.joblib")
MODEL_INFO_PATH = os.environ.get("MODEL_INFO_PATH", "model/model_info.json")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("api")
warnings.filterwarnings("ignore", message="X does not have valid feature names")

state = {"model": None, "info": {}}
LABELS = {0: "malignant", 1: "benign"}


@asynccontextmanager
async def lifespan(app):
    """Load the model once at startup. If that fails, the app stops with an error."""
    log.info("starting app version %s, loading model from %s", APP_VERSION, MODEL_PATH)
    try:
        state["model"] = joblib.load(MODEL_PATH)
        with open(MODEL_INFO_PATH) as f:
            state["info"] = json.load(f)
    except Exception as e:
        log.error("FAILED to load model from %s: %s", MODEL_PATH, e)
        raise
    log.info("model loaded: %s version %s", state["info"].get("model_name"),
             state["info"].get("model_version"))
    yield


app = FastAPI(title="Cancer prediction API", version=APP_VERSION, lifespan=lifespan)


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready"}


@app.get("/version")
def version():
    return {"app_version": APP_VERSION, "model": state["info"]}


@app.post("/predict")
def predict(req: PredictRequest):
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    if len(req.features) != model.n_features_in_:
        raise HTTPException(
            status_code=422,
            detail=f"expected {model.n_features_in_} features, got {len(req.features)}",
        )
    x = np.array([req.features])
    label = int(model.predict(x)[0])
    probability = float(model.predict_proba(x)[0][label])
    log.info("prediction=%s probability=%.4f", LABELS[label], probability)
    return {
        "prediction": label,
        "label": LABELS[label],
        "probability": round(probability, 4),
        "model_version": state["info"].get("model_version"),
    }
