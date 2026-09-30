# Weather Intelligence API

**AT2 – Machine Learning as a Service** · 36120 Advanced Machine Learning Applications
**Student:** Tahreema Rahman Zariyat · **Student ID:** 26016360

FastAPI service that serves the two models trained in the
[experimentation repository](https://github.com/tahreemarz/weather-intelligence-experiments)
for Sydney, NSW (lat −33.8688, lon 151.2093):

| Endpoint | Model | Output |
| --- | --- | --- |
| `GET /predict/index/comfort_climate?date=YYYY-MM-DD` | XGBoost regressors (one per horizon) | Climate Comfort Index (0–100) for `date+1`, `date+2`, `date+3` |
| `GET /predict/category/weather_hazard?date=YYYY-MM-DD` | XGBoost multiclass classifier | Weather Hazard Category for `date+7` |

Live deployment: **https://weather-intelligence-api.onrender.com** *(update after deploying —
free instances sleep after inactivity, so the first request can take ~1 minute)*.

All modelling logic — Open-Meteo access, daily aggregation, cleaning, the target
formulas, feature engineering, artefact loading and prediction — comes from my custom
package **`amla-package-26016360` ≥ 0.2.0** (TestPyPI), the same code the notebooks use.
The API itself is ~250 lines of routing, validation and error handling.

## Repository structure

```
├── README.md
├── Dockerfile               <- python:3.11-slim image, installs the package from TestPyPI, runs uvicorn
├── render.yaml              <- Render blueprint (docker runtime, /health check)
├── pyproject.toml           <- dependencies (custom package + FastAPI stack)
├── requirements.txt         <- public dependencies (used by the Dockerfile; the custom package is added from TestPyPI)
├── app
│   ├── __init__.py
│   └── main.py              <- FastAPI application
├── models
│   ├── comfort_climate      <- cci_h1.json, cci_h2.json, cci_h3.json, climatology.json, metadata.json
│   └── weather_hazard       <- whc_h7.json, climatology.json, metadata.json
└── tests
    ├── conftest.py          <- trains tiny bundles on synthetic data, mocks Open-Meteo
    └── test_api.py          <- 17 contract tests (responses, formats, error codes)
```

The artefacts in `models/` are produced by the experimentation notebooks
(`models/<target>/` there) and copied here unchanged. They are plain JSON (XGBoost's
portable format + climatology table + metadata), so no pickle / version coupling.

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/` | Project name, description, version, GitHub link, list of endpoints, expected input/output formats |
| GET | `/health` | `{"status": "healthy", "message": "Weather Intelligence API is running"}` |
| GET | `/predict/index/comfort_climate?date=YYYY-MM-DD` | CCI forecast for the three days after `date` |
| GET | `/predict/category/weather_hazard?date=YYYY-MM-DD` | Hazard category exactly seven days after `date` |
| GET | `/model-metadata` | List with one entry per model: target, prediction type, algorithm, forecast horizon, features, feature importance, hyper-parameters, validation/test performance, baselines, training window |
| GET | `/docs` | Interactive OpenAPI documentation |

### Example responses

```
GET /predict/index/comfort_climate?date=2025-12-15
{
  "input_date": "2025-12-15",
  "predictions": {
    "comfort_climate": {"2025-12-16": 71.3, "2025-12-17": 68.9, "2025-12-18": 66.0}
  }
}

GET /predict/category/weather_hazard?date=2025-12-15
{
  "input_date": "2025-12-15",
  "predictions": {"weather_hazard": {"2025-12-22": "Moderate Risk"}}
}
```

### How a prediction is produced

1. `date` is validated (format, real calendar date, not later than yesterday in Sydney
   time, not before 1940-04-01).
2. The 90 days of daily observations ending on `date` are fetched from the Open-Meteo
   Historical Weather API (`fetch_daily_window`), with the most recent days back-filled
   from the Forecast API when the archive has not caught up, then cleaned with the same
   package functions as in training (`clean_daily`: incomplete-day masking, physical
   range validation, short-gap interpolation).
3. The feature row for `date` is built with the package's `prepare_features` (lags,
   rolling statistics, tendencies, seasonality, climatology) and ordered with the
   frozen feature list stored in the artefact.
4. The model(s) predict; the CCI is clipped to 0–100 and rounded to one decimal, the
   hazard class index is mapped to its label.

Windows for dates older than ten days are cached in memory (archive data is final).

### Error handling

| Situation | Status | Body |
| --- | --- | --- |
| `date` missing | 422 | `{"error": "Invalid request", "detail": [...], "hint": ...}` |
| `date` malformed (`15/12/2025`, `2025-02-30`, …) | 422 | `{"error": "Invalid date ...: expected the format YYYY-MM-DD ..."}` |
| `date` today / future | 400 | `{"error": "Date ... is in the future or is today: observations are only available up to ..."}` |
| `date` before 1940-04-01 | 400 | `{"error": "Date ... is too early ..."}` |
| No complete observation yet for `date` (Open-Meteo lags 1–2 days) | 503 | `{"error": "Weather observations are not available for ...: ..."}` |
| Open-Meteo unreachable | 503 | `{"error": "The Open-Meteo API could not be reached ..."}` |
| Model artefacts missing on the server | 503 | `{"error": "Model '...' is not available on this server ..."}` |
| Unknown route | 404 | `{"error": "Unknown endpoint '/...'. Available endpoints: ..."}` |
| Unexpected failure | 500 | `{"error": "Internal server error while producing the prediction."}` |

## Run locally

```bash
git clone https://github.com/tahreemarz/weather-intelligence-api.git
cd weather-intelligence-api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # public dependencies from PyPI
pip install --no-deps --index-url https://test.pypi.org/simple/ "amla-package-26016360>=0.2.0"   # custom package
pip install -e ".[dev]"                    # pytest + httpx for the tests

pytest                                     # 17 tests, no network needed
uvicorn app.main:app --reload --port 8000  # then open http://127.0.0.1:8000/docs
```

Make sure `models/comfort_climate/` and `models/weather_hazard/` contain the artefacts
from the experimentation repository (copy `models/*` from there). Without them the API
still starts; `/health` answers and the prediction endpoints return 503 with an
explanation.

### Docker

```bash
docker build -t weather-intelligence-api .
docker run --rm -p 8000:8000 weather-intelligence-api
curl "http://localhost:8000/predict/index/comfort_climate?date=2025-12-15"
```

## Deploy on Render

1. Push this repository (including `models/`) to a **private GitHub repository**.
2. On Render: **New + → Blueprint**, connect the repository; Render reads `render.yaml`
   (Docker runtime, free plan, health check on `/health`). Alternatively **New + → Web
   Service**, runtime *Docker*, health check path `/health`, no build/start commands
   needed (they come from the Dockerfile).
3. Optional environment variables (already set in `render.yaml`): `GITHUB_URL`,
   `EXPERIMENTS_URL`, `LOG_LEVEL`, `MODELS_DIR` (default `/app/models`).
4. Deploy. The first build takes a few minutes (installing xgboost / scikit-learn).
   Verify with `https://<service>.onrender.com/health` and `/docs`.
5. `autoDeploy: true` redeploys on every push to the default branch, e.g. after copying
   retrained artefacts into `models/`.

Notes for the free plan: the instance sleeps after 15 min without traffic (first
request afterwards takes up to a minute), and memory is 512 MB, which is sufficient
(the service uses ~250 MB with both bundles loaded).

## Updating the models or the package

* **New artefacts**: re-run the notebooks in the experimentation repository, copy
  `models/comfort_climate` and `models/weather_hazard` here, commit, push.
* **Package change** (formulas, features, cleaning): change it in the
  [package repository](https://github.com/tahreemarz/36120-26SP-group2-26016360-package),
  bump the version, `python -m build && twine upload --repository testpypi dist/*`, then
  bump the pin in `requirements.txt` / `pyproject.toml` in both repositories and retrain,
  because the frozen feature list inside the artefacts must match the package version
  that computes the features.
