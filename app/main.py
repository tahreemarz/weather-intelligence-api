"""Weather Intelligence API - FastAPI service serving the two Sydney forecasting models.

Endpoints (all GET):

* ``/``                                   project information and endpoint list
* ``/health``                             liveness probe
* ``/predict/index/comfort_climate``      Climate Comfort Index for the 3 days after ``date``
* ``/predict/category/weather_hazard``    Weather Hazard Category exactly 7 days after ``date``
* ``/model-metadata``                     target, algorithm, horizon, features, importance, metrics

Every piece of modelling logic (Open-Meteo access, daily aggregation, cleaning, target
formulas, feature engineering, artefact loading, prediction) comes from the custom
package ``amla_package_26016360`` (>= 0.2.0), exactly as used by the training notebooks.
"""

from __future__ import annotations

import logging
import os
import re
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

import amla_package_26016360 as amla
from amla_package_26016360.weather import (
    SYDNEY,
    DataUnavailableError,
    ModelBundle,
    clean_daily,
    fetch_daily_window,
    predict_cci,
    predict_whc,
)

logger = logging.getLogger("weather_intelligence_api")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

APP_VERSION = "1.0.0"
PROJECT_NAME = "Weather Intelligence API"
DESCRIPTION = (
    "Machine-learning weather intelligence services for Sydney, NSW (lat -33.8688, lon 151.2093): "
    "a Climate Comfort Index forecast for the next 3 days and a Weather Hazard Category forecast "
    "for exactly 7 days ahead, both trained on Open-Meteo historical observations."
)
GITHUB_URL = os.getenv("GITHUB_URL", "https://github.com/tahreemarz/weather-intelligence-api")
EXPERIMENTS_URL = os.getenv("EXPERIMENTS_URL", "https://github.com/tahreemarz/weather-intelligence-experiments")
MODELS_DIR = Path(os.getenv("MODELS_DIR", Path(__file__).resolve().parent.parent / "models"))
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "90"))          # observation window needed for the features
EARLIEST_DATE = date(1940, 1, 1) + timedelta(days=LOOKBACK_DAYS)   # first day with a full feature window (ERA5 starts 1940)
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

ENDPOINTS = [
    {"path": "/", "method": "GET", "description": "Project information and list of endpoints"},
    {"path": "/health", "method": "GET", "description": "Health check"},
    {"path": "/predict/index/comfort_climate", "method": "GET", "params": {"date": "YYYY-MM-DD"},
     "description": "Climate Comfort Index (0-100) forecast for the 3 days following the input date"},
    {"path": "/predict/category/weather_hazard", "method": "GET", "params": {"date": "YYYY-MM-DD"},
     "description": "Weather Hazard Category (Low / Moderate / High / Extreme Risk) exactly 7 days after the input date"},
    {"path": "/model-metadata", "method": "GET", "description": "Metadata of the deployed models (target, algorithm, horizon, features, importance, performance)"},
    {"path": "/docs", "method": "GET", "description": "Interactive OpenAPI documentation"},
]


# ---------------------------------------------------------------------------
# Model registry (loaded once at start-up)
# ---------------------------------------------------------------------------
class ModelRegistry:
    def __init__(self) -> None:
        self.bundles: dict[str, ModelBundle] = {}
        self.errors: dict[str, str] = {}
        self._cache: OrderedDict[str, Any] = OrderedDict()   # daily windows for "final" dates (archive data never changes)

    def load(self, models_dir: Path) -> None:
        for name in ("comfort_climate", "weather_hazard"):
            path = models_dir / name
            try:
                self.bundles[name] = ModelBundle.load(path)
                logger.info("loaded model bundle '%s' from %s (%d features)", name, path, len(self.bundles[name].features))
            except Exception as exc:  # noqa: BLE001 - keep the API up, report through /model-metadata and 503s
                self.errors[name] = f"{type(exc).__name__}: {exc}"
                logger.error("could not load model bundle '%s' from %s: %s", name, path, exc)

    def get(self, name: str) -> ModelBundle:
        if name not in self.bundles:
            raise HTTPException(
                status_code=503,
                detail=f"Model '{name}' is not available on this server ({self.errors.get(name, 'not loaded')}). "
                       "Copy the artefacts from the experimentation repository into the models/ folder and restart.",
            )
        return self.bundles[name]

    # -- observation window -------------------------------------------------
    def daily_window(self, input_date: date):
        key = input_date.isoformat()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        window = clean_daily(fetch_daily_window(input_date, lookback_days=LOOKBACK_DAYS, location=SYDNEY))
        if input_date <= today_in_sydney() - timedelta(days=10):      # archive data for old dates is final -> cacheable
            self._cache[key] = window
            if len(self._cache) > 256:
                self._cache.popitem(last=False)
        return window


registry = ModelRegistry()


def today_in_sydney() -> date:
    return datetime.now(ZoneInfo(SYDNEY.timezone)).date()


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry.load(MODELS_DIR)
    yield


app = FastAPI(
    title=PROJECT_NAME,
    description=DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
    contact={"name": "Tahreema Rahman Zariyat (26016360)", "url": GITHUB_URL},
)


# ---------------------------------------------------------------------------
# Error handling: always JSON, always an explanation
# ---------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    problems = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err.get("loc", []) if x not in ("query", "path", "body"))
        problems.append(f"{loc}: {err.get('msg')}")
    return JSONResponse(status_code=422, content={
        "error": "Invalid request",
        "detail": problems,
        "hint": "Dates must be given as ?date=YYYY-MM-DD, e.g. /predict/index/comfort_climate?date=2025-12-15",
    })


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    logger.exception("unexpected error on %s", request.url)
    return JSONResponse(status_code=500, content={"error": "Internal server error while producing the prediction.",
                                                  "detail": f"{type(exc).__name__}"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_input_date(raw: str) -> date:
    """Validate the ``date`` query parameter and return it as a ``date``."""
    if not DATE_RE.match(raw):
        raise HTTPException(status_code=422, detail=f"Invalid date '{raw}': expected the format YYYY-MM-DD (e.g. 2025-12-15).")
    try:
        parsed = date.fromisoformat(raw)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid date '{raw}': not a real calendar date.")
    yesterday = today_in_sydney() - timedelta(days=1)
    if parsed > yesterday:
        raise HTTPException(
            status_code=400,
            detail=f"Date {raw} is in the future or is today: observations are only available up to {yesterday.isoformat()} "
                   f"(Sydney time). The model forecasts from observed days, so please use a date <= {yesterday.isoformat()}.",
        )
    if parsed < EARLIEST_DATE:
        raise HTTPException(status_code=400, detail=f"Date {raw} is too early: observations start in 1940 and {LOOKBACK_DAYS} days "
                                                    f"of history are needed, so the earliest supported date is {EARLIEST_DATE.isoformat()}.")
    return parsed


def observation_window(input_date: date):
    """Fetch (and clean) the trailing daily observation window, mapping upstream problems to HTTP errors."""
    try:
        return registry.daily_window(input_date)
    except DataUnavailableError as exc:
        raise HTTPException(status_code=503, detail=f"Weather observations are not available for {input_date.isoformat()}: {exc}")
    except requests.RequestException as exc:
        raise HTTPException(status_code=503, detail=f"The Open-Meteo API could not be reached ({type(exc).__name__}). Please retry shortly.")


def public_metadata(bundle: ModelBundle) -> dict[str, Any]:
    meta = bundle.metadata
    return {
        "target": meta.get("target"),
        "target_code": meta.get("target_code"),
        "prediction_type": meta.get("prediction_type"),
        "classes": meta.get("classes"),
        "output_range": meta.get("output_range"),
        "algorithm": meta.get("algorithm"),
        "forecast_horizon": meta.get("forecast_horizon"),
        "horizons": meta.get("horizons"),
        "n_features": meta.get("n_features", len(bundle.features)),
        "features": bundle.features,
        "feature_importance": meta.get("feature_importance"),
        "hyperparameters": meta.get("hyperparameters"),
        "performance": meta.get("performance"),
        "training_data": meta.get("training_data"),
        "library_versions": meta.get("library_versions"),
        "package_version": meta.get("package_version"),
        "saved_at": meta.get("saved_at"),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["info"])
def root():
    return {
        "project": PROJECT_NAME,
        "description": DESCRIPTION,
        "version": APP_VERSION,
        "github": GITHUB_URL,
        "experimentation_repository": EXPERIMENTS_URL,
        "package": f"amla-package-26016360 {amla.__version__} (TestPyPI)",
        "location": {"name": SYDNEY.name, "latitude": SYDNEY.latitude, "longitude": SYDNEY.longitude, "timezone": SYDNEY.timezone},
        "models_loaded": sorted(registry.bundles),
        "endpoints": ENDPOINTS,
        "input_format": {"date": "YYYY-MM-DD, any observed day up to yesterday (Sydney time)"},
        "output_format": {
            "/predict/index/comfort_climate": {"input_date": "YYYY-MM-DD", "predictions": {"comfort_climate": {"date+1": "float", "date+2": "float", "date+3": "float"}}},
            "/predict/category/weather_hazard": {"input_date": "YYYY-MM-DD", "predictions": {"weather_hazard": {"date+7": "Low Risk | Moderate Risk | High Risk | Extreme Risk"}}},
        },
    }


@app.get("/health", tags=["info"])
def health():
    return {"status": "healthy", "message": "Weather Intelligence API is running"}


@app.get("/predict/index/comfort_climate", tags=["predictions"])
def predict_comfort_climate(date: str = Query(..., description="Input date, YYYY-MM-DD (observed day, up to yesterday Sydney time)")):
    input_date = parse_input_date(date)
    bundle = registry.get("comfort_climate")
    window = observation_window(input_date)
    try:
        forecast = predict_cci(bundle, window, input_date)
    except KeyError as exc:
        raise HTTPException(status_code=503, detail=f"Could not build the model inputs for {date}: {exc}")
    return {"input_date": input_date.isoformat(), "predictions": {"comfort_climate": forecast}}


@app.get("/predict/category/weather_hazard", tags=["predictions"])
def predict_weather_hazard(date: str = Query(..., description="Input date, YYYY-MM-DD (observed day, up to yesterday Sydney time)")):
    input_date = parse_input_date(date)
    bundle = registry.get("weather_hazard")
    window = observation_window(input_date)
    try:
        result = predict_whc(bundle, window, input_date)
    except KeyError as exc:
        raise HTTPException(status_code=503, detail=f"Could not build the model inputs for {date}: {exc}")
    return {"input_date": input_date.isoformat(), "predictions": {"weather_hazard": {result["date"]: result["label"]}}}


@app.get("/model-metadata", tags=["info"])
def model_metadata():
    items = [public_metadata(registry.bundles[name]) for name in ("comfort_climate", "weather_hazard") if name in registry.bundles]
    if not items:
        raise HTTPException(status_code=503, detail=f"No model artefacts are loaded ({registry.errors}).")
    return items


@app.get("/{path:path}", include_in_schema=False)
def not_found(path: str):
    raise HTTPException(status_code=404, detail=f"Unknown endpoint '/{path}'. Available endpoints: " + ", ".join(e["path"] for e in ENDPOINTS))
