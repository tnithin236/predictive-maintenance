"""Pydantic request / response models. Ranges are sanity bounds, not learned limits."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class SensorReading(BaseModel):
    """One machine reading using the AI4I 2020 sensor set."""

    model_config = ConfigDict(json_schema_extra={"example": {
        "machine_id": "M14860", "machine_type": "M", "air_temperature_k": 298.1,
        "process_temperature_k": 308.6, "rotational_speed_rpm": 1551, "torque_nm": 42.8,
        "tool_wear_min": 0}})

    machine_id: Optional[str] = Field(None, max_length=50)
    machine_type: Literal["L", "M", "H"]
    air_temperature_k: float = Field(..., gt=200, lt=450)
    process_temperature_k: float = Field(..., gt=200, lt=450)
    rotational_speed_rpm: float = Field(..., gt=0, le=10000)
    torque_nm: float = Field(..., ge=0, le=500)
    tool_wear_min: float = Field(..., ge=0, le=2000)


class FactorDetail(BaseModel):
    feature: str
    contribution: float
    value: Optional[float | str] = None
    direction: Literal["increases_failure_risk", "decreases_failure_risk"]


class PredictionResponse(BaseModel):
    machine_id: Optional[str] = None
    prediction: Literal[0, 1]
    failure_probability: float = Field(..., ge=0, le=1)
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    anomaly_score: float
    anomaly_status: Literal["NORMAL", "ANOMALY"]
    top_factors: list[str]
    factor_details: list[FactorDetail] = []
    model_version: str
    note: str = ("anomaly_score is NOT a failure probability; factors are model-derived "
                 "contributions, not guaranteed causes.")


class BatchRequest(BaseModel):
    readings: list[SensorReading] = Field(..., min_length=1)


class BatchResponse(BaseModel):
    count: int
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    detail: Optional[str] = None
