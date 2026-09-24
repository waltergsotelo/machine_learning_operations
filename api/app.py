import json
import os
from pathlib import Path
from threading import Lock
from typing import Any

import mlflow
import mlflow.artifacts
import mlflow.pytorch
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

MODEL_NAME = os.getenv("MODEL_NAME", "purchase-spend-regressor")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        description="Features numéricas con los nombres expuestos por /model-info."
    )


class ModelService:
    def __init__(self) -> None:
        self.model: torch.jit.ScriptModule | None = None
        self.feature_columns: list[str] = []
        self.model_version: str | None = None
        self.lock = Lock()

    def load(self) -> None:
        with self.lock:
            client = MlflowClient()
            model_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
            schema_path = mlflow.artifacts.download_artifacts(
                run_id=model_version.run_id,
                artifact_path="feature_schema.json",
            )
            schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
            self.feature_columns = schema["feature_columns"]
            self.model = mlflow.pytorch.load_model(MODEL_URI)
            self.model.eval()
            self.model_version = model_version.version

    def predict(self, features: dict[str, float]) -> float:
        if self.model is None:
            raise RuntimeError("El modelo aún no fue cargado.")
        expected_columns = set(self.feature_columns)
        provided_columns = set(features)
        missing_columns = sorted(expected_columns - provided_columns)
        unexpected_columns = sorted(provided_columns - expected_columns)
        if missing_columns or unexpected_columns:
            details = []
            if missing_columns:
                details.append(f"Faltan features: {', '.join(missing_columns)}.")
            if unexpected_columns:
                details.append(f"Features desconocidas: {', '.join(unexpected_columns)}.")
            raise ValueError(
                " ".join(details)
            )
        ordered_features = [features[column] for column in self.feature_columns]
        input_tensor = torch.tensor(np.asarray([ordered_features], dtype=np.float32))
        with torch.no_grad():
            prediction = self.model(input_tensor)
        return float(prediction.squeeze().item())


model_service = ModelService()
app = FastAPI(title="Purchase Spend Predictor", version="0.1.0")


@app.on_event("startup")
def load_production_model() -> None:
    model_service.load()


@app.get("/health")
def health() -> dict[str, Any]:
    if model_service.model is None:
        raise HTTPException(status_code=503, detail="El modelo no está disponible.")
    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "model_version": model_service.model_version,
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    if model_service.model is None:
        raise HTTPException(status_code=503, detail="El modelo no está disponible.")
    return {
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "model_version": model_service.model_version,
        "feature_columns": model_service.feature_columns,
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    try:
        value = model_service.predict(request.features)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {
        "prediction": value,
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "model_version": model_service.model_version,
    }
