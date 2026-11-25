"""FastAPI inference service for validated OHLC history."""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from olm import __version__
from olm.config import load_config
from olm.data.validation import validate_frame
from olm.inference.forecast import Forecaster, derive_context_frames
from olm.monitoring.prediction_log import PredictionLog


class Candle(BaseModel):
    timestamp: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)


class ForecastRequest(BaseModel):
    # 192 M5 bars of context plus enough history to derive 16 H4 bars (768 M5).
    m5: list[Candle] = Field(min_length=1000, max_length=20000)
    scenarios: int = Field(default=32, ge=1, le=512)
    seed: int | None = None


class ActualRequest(BaseModel):
    forecast_id: str = Field(min_length=8, max_length=64)
    candles: list[Candle] = Field(min_length=3, max_length=3)


class ServiceState:
    def __init__(self) -> None:
        self.forecaster: Forecaster | None = None
        self.checkpoint = os.environ.get('OLM_CHECKPOINT', load_config().inference.checkpoint)
        self.config = load_config()
        self.log = PredictionLog(self.config.inference.prediction_log)

    def get(self) -> Forecaster:
        if self.forecaster is None:
            if not os.path.exists(self.checkpoint):
                raise HTTPException(status_code=503, detail=f'checkpoint not found: {self.checkpoint}')
            self.forecaster = Forecaster(self.checkpoint, self.config)
        return self.forecaster


state = ServiceState()
app = FastAPI(title='OLM-Gold', version=__version__, description='XAU/USD probabilistic OHLC forecasting')


@app.get('/health')
def health() -> dict:
    return {
        'status': 'ok',
        'version': __version__,
        'checkpoint': state.checkpoint,
        'checkpoint_loaded': state.forecaster is not None,
    }


@app.get('/v1/model')
def model_info() -> dict:
    forecaster = state.get()
    return {
        'model_version': forecaster.model_version,
        'parameters': forecaster.model.parameter_count(),
        'temporal_dim': forecaster.model.temporal_dim,
        'contexts': {
            'm5': forecaster.model_config.m5_context,
            'm15': forecaster.model_config.m15_context,
            'h1': forecaster.model_config.h1_context,
            'h4': forecaster.model_config.h4_context,
        },
        'scenarios_default': forecaster.model_config.n_scenarios,
    }


@app.post('/v1/forecast')
def forecast(request: ForecastRequest) -> dict:
    frame = pd.DataFrame([c.model_dump() for c in request.m5])
    report = validate_frame(frame, 5)
    if not report.ok:
        raise HTTPException(status_code=422, detail=report.errors)
    try:
        frames = derive_context_frames(frame)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    forecaster = state.get()
    try:
        result = forecaster.forecast(frames, scenarios=request.scenarios, seed=request.seed)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    payload = result.to_dict()
    record = state.log.record_forecast(payload, forecaster.model_version, result.as_of)
    payload['forecast_id'] = record['forecast_id']
    payload['model_version'] = forecaster.model_version
    payload['disclaimer'] = 'Probabilistic research output; not investment advice.'
    return payload


@app.post('/v1/actual')
def actual(request: ActualRequest) -> dict:
    candles = [c.model_dump() for c in request.candles]
    try:
        evaluation = state.log.evaluate_forecast(request.forecast_id, candles)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    state.log.record_actual(request.forecast_id, candles)
    return evaluation


@app.get('/v1/predictions/{forecast_id}')
def prediction(forecast_id: str) -> dict:
    for entry in state.log.read_all():
        if entry.get('forecast_id') == forecast_id and entry.get('type') == 'forecast':
            return entry
    raise HTTPException(status_code=404, detail='unknown forecast id')
