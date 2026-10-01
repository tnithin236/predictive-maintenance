"""FastAPI prediction service. Loads a trained bundle; never trains at startup."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from api.schemas import BatchRequest, BatchResponse, HealthResponse, PredictionResponse, SensorReading
from src import config
from src.predict import Predictor

logger = logging.getLogger("api")


def create_app(model_path: Path | str | None = None) -> FastAPI:
    """Application factory (lets tests point the API at a temporary model)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = None
        app.state.load_error = None
        try:
            app.state.predictor = Predictor.from_path(model_path)
            logger.info("Loaded model %s", app.state.predictor.bundle["model_version"])
        except Exception as exc:  # noqa: BLE001 - keep API up so /health can report why
            app.state.load_error = str(exc)
            logger.exception("Model failed to load")
        yield

    app = FastAPI(title="Predictive Maintenance API", version="1.0.0", lifespan=lifespan)

    def predictor_of(request: Request) -> Predictor:
        predictor = request.app.state.predictor
        if predictor is None:
            raise HTTPException(status_code=503, detail=f"Model not loaded: {request.app.state.load_error}")
        return predictor

    def log_to_db(results: list[dict[str, Any]]) -> None:
        if not config.DATABASE_URL:
            return
        try:
            from src.database import save_predictions

            save_predictions(results)
        except Exception:  # noqa: BLE001 - DB logging must never break predictions
            logger.exception("Could not persist predictions to the database")

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def unhandled_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error", exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        loaded = request.app.state.predictor is not None
        return HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded,
                              detail=request.app.state.load_error)

    @app.get("/model/info")
    def model_info(request: Request) -> dict[str, Any]:
        return predictor_of(request).model_info()

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payload: SensorReading, request: Request) -> dict[str, Any]:
        results = predictor_of(request).predict([payload.model_dump()], explain=True)
        log_to_db(results)
        return results[0]

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(payload: BatchRequest, request: Request, explain: bool = False) -> dict[str, Any]:
        if len(payload.readings) > config.MAX_BATCH_SIZE:
            raise HTTPException(status_code=413, detail=f"Batch too large (max {config.MAX_BATCH_SIZE}).")
        results = predictor_of(request).predict([r.model_dump() for r in payload.readings], explain=explain)
        log_to_db(results)
        return {"count": len(results), "predictions": results}

    return app


app = create_app()
