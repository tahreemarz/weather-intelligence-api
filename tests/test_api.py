"""API contract tests (run with `pytest`; no network access needed)."""

from datetime import date, timedelta

import pytest


def test_root_lists_project_info_and_endpoints(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    for key in ("project", "description", "version", "github", "endpoints"):
        assert key in body
    paths = {e["path"] for e in body["endpoints"]}
    assert {"/", "/health", "/predict/index/comfort_climate", "/predict/category/weather_hazard", "/model-metadata"} <= paths


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy", "message": "Weather Intelligence API is running"}


def test_comfort_climate_prediction_shape(client, valid_date):
    r = client.get("/predict/index/comfort_climate", params={"date": valid_date})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["input_date"] == valid_date
    preds = body["predictions"]["comfort_climate"]
    d = date.fromisoformat(valid_date)
    assert list(preds) == [(d + timedelta(days=h)).isoformat() for h in (1, 2, 3)]
    assert all(isinstance(v, float) and 0 <= v <= 100 for v in preds.values())


def test_weather_hazard_prediction_shape(client, valid_date):
    r = client.get("/predict/category/weather_hazard", params={"date": valid_date})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["input_date"] == valid_date
    preds = body["predictions"]["weather_hazard"]
    d = date.fromisoformat(valid_date)
    assert list(preds) == [(d + timedelta(days=7)).isoformat()]
    assert list(preds.values())[0] in {"Low Risk", "Moderate Risk", "High Risk", "Extreme Risk"}


def test_predictions_are_deterministic(client, valid_date):
    a = client.get("/predict/index/comfort_climate", params={"date": valid_date}).json()
    b = client.get("/predict/index/comfort_climate", params={"date": valid_date}).json()
    assert a == b


def test_model_metadata(client):
    r = client.get("/model-metadata")
    assert r.status_code == 200
    items = r.json()
    assert isinstance(items, list) and len(items) == 2
    for item in items:
        for key in ("target", "prediction_type", "algorithm", "forecast_horizon", "features", "feature_importance", "performance"):
            assert key in item, key
    assert {i["prediction_type"] for i in items} == {"regression", "classification"}


@pytest.mark.parametrize("endpoint", ["/predict/index/comfort_climate", "/predict/category/weather_hazard"])
def test_missing_date_is_422(client, endpoint):
    r = client.get(endpoint)
    assert r.status_code == 422
    assert "error" in r.json()


@pytest.mark.parametrize("bad", ["2025-13-01", "2025-02-30", "15/12/2025", "yesterday", "2025-1-5"])
def test_malformed_date_is_422(client, bad):
    r = client.get("/predict/index/comfort_climate", params={"date": bad})
    assert r.status_code == 422
    assert "error" in r.json()


def test_future_date_is_400(client):
    tomorrow = (date.today() + timedelta(days=2)).isoformat()
    r = client.get("/predict/category/weather_hazard", params={"date": tomorrow})
    assert r.status_code == 400
    assert "future" in r.json()["error"].lower() or "today" in r.json()["error"].lower()


def test_too_early_date_is_400(client):
    r = client.get("/predict/index/comfort_climate", params={"date": "1930-01-01"})
    assert r.status_code == 400


def test_unavailable_observation_is_503(client, valid_date):
    # A valid past date for which the (mocked) upstream has no observation yet.
    after = (date.fromisoformat(valid_date) + timedelta(days=3)).isoformat()
    r = client.get("/predict/index/comfort_climate", params={"date": after})
    assert r.status_code == 503
    assert "not available" in r.json()["error"]


def test_unknown_route_is_404_json(client):
    r = client.get("/predict/something")
    assert r.status_code == 404
    assert "Unknown endpoint" in r.json()["error"]
