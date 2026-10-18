import json
import math
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import mlflow
import mlflow.artifacts
import mlflow.pytorch
import numpy as np
import torch
from fastapi import FastAPI, Header, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

MODEL_NAME = os.getenv("MODEL_NAME", "purchase-spend-regressor")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")
# Color de esta instancia en el deploy Blue-Green ("blue" o "green").
API_COLOR = os.getenv("API_COLOR", "")
# Token que exige /admin/load. Si está vacío, el endpoint queda deshabilitado.
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        description="Features numéricas con los nombres expuestos por /model-info."
    )


class LoadRequest(BaseModel):
    version: str = Field(description="Versión registrada en MLflow que se va a cargar.")


@dataclass(frozen=True)
class LoadedModel:
    """Una versión del modelo lista para predecir, con su esquema de features."""

    version: str
    feature_columns: list[str]
    model: torch.jit.ScriptModule

    def predict(self, features: dict[str, float]) -> float:
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


class ModelService:
    """Mantiene el modelo en servicio y permite reemplazarlo sin cortar predicciones.

    Las predicciones leen self.current una sola vez, así que reemplazar esa
    referencia es atómico: cada petición usa el modelo anterior o el nuevo,
    nunca uno a medio cargar.
    """

    def __init__(self) -> None:
        self.current: LoadedModel | None = None
        self.lock = Lock()

    def load_alias(self) -> None:
        """Carga la versión a la que apunta MODEL_ALIAS; falla si el alias no existe."""
        model_version = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        self.load_version(model_version.version)

    def load_version(self, version: str) -> str | None:
        """Carga y valida una versión antes de ponerla en servicio.

        Devuelve la versión que estaba en servicio. Si la carga o la validación
        fallan, se propaga el error y sigue en servicio el modelo anterior.
        """
        with self.lock:
            previous = self.current
            if previous is not None and previous.version == version:
                return previous.version
            candidate = self._load(version)
            smoke_prediction = candidate.predict(
                {column: 0.0 for column in candidate.feature_columns}
            )
            if not math.isfinite(smoke_prediction):
                raise ValueError(f"La versión {version} devolvió una predicción no finita.")
            self.current = candidate
            return previous.version if previous is not None else None

    @staticmethod
    def _load(version: str) -> LoadedModel:
        model_version = MlflowClient().get_model_version(MODEL_NAME, version)
        schema_path = mlflow.artifacts.download_artifacts(
            run_id=model_version.run_id,
            artifact_path="feature_schema.json",
        )
        schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        model = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}/{version}")
        model.eval()
        return LoadedModel(
            version=model_version.version,
            feature_columns=schema["feature_columns"],
            model=model,
        )


model_service = ModelService()
app = FastAPI(title="Purchase Spend Predictor", version="0.1.0")


def require_model() -> LoadedModel:
    current = model_service.current
    if current is None:
        raise HTTPException(status_code=503, detail="El modelo no está disponible.")
    return current


@app.on_event("startup")
def load_production_model() -> None:
    model_service.load_alias()


@app.get("/health")
def health() -> dict[str, Any]:
    current = require_model()
    return {
        "status": "ok",
        "color": API_COLOR,
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "model_version": current.version,
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    current = require_model()
    return {
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "model_version": current.version,
        "feature_columns": current.feature_columns,
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    current = require_model()
    try:
        value = current.predict(request.features)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {
        "prediction": value,
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "model_version": current.version,
    }


@app.post("/admin/load", include_in_schema=False)
def admin_load(
    request: LoadRequest, x_admin_token: str = Header(default="")
) -> dict[str, Any]:
    """Carga una versión en esta instancia. Lo usa el DAG deploy_purchase_model.

    No se publica a través del proxy: solo es accesible desde la red interna.
    """
    if not ADMIN_TOKEN or not secrets.compare_digest(x_admin_token, ADMIN_TOKEN):
        raise HTTPException(status_code=403, detail="Token de administración inválido.")
    try:
        previous_version = model_service.load_version(request.version)
    except Exception as error:
        serving = model_service.current.version if model_service.current else None
        raise HTTPException(
            status_code=500,
            detail=(
                f"No se pudo cargar la versión {request.version}: {error}. "
                f"Sigue en servicio la versión {serving}."
            ),
        ) from error
    return {
        "color": API_COLOR,
        "previous_version": previous_version,
        "model_version": request.version,
    }
