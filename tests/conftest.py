"""Fixtures: tiny model bundles trained on synthetic data (no network) and a mocked Open-Meteo window."""

from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd
import pytest

from amla_package_26016360.weather import (
    Climatology,
    ModelBundle,
    add_future_targets,
    add_targets,
    aggregate_daily,
    balanced_sample_weights,
    clean_daily,
    fit_xgb_classifier,
    fit_xgb_regressor,
    observation_columns,
    prepare_features,
)
from amla_package_26016360.weather.synthetic import make_synthetic_hourly

# Dates of the synthetic history: must end before "yesterday" in Sydney so requests are valid.
HIST_START, HIST_END = "2018-01-01", (date.today() - timedelta(days=30)).isoformat()


@pytest.fixture(scope="session")
def synthetic_daily():
    daily_raw = aggregate_daily(make_synthetic_hourly(HIST_START, HIST_END, seed=3))
    return clean_daily(daily_raw)


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory, synthetic_daily):
    """Train minimal bundles for both targets and save them as the API expects."""
    root = tmp_path_factory.mktemp("models")
    daily = synthetic_daily
    df = add_future_targets(add_targets(daily))
    clim = Climatology().fit(df)
    obs = daily[observation_columns(daily) + ["n_hours"]]
    feats = prepare_features(obs, clim)
    features = [c for c in feats.columns if c not in ("n_hours", "whc") and not c.endswith(("_score", "_hazard"))]
    ok = df[["cci_h1", "cci_h2", "cci_h3", "whc_h7"]].notna().all(axis=1) & (feats.index >= pd.Timestamp(HIST_START) + pd.Timedelta(days=30))
    X, y = feats.loc[ok, features], df.loc[ok]

    cci_models = {f"cci_h{h}": fit_xgb_regressor(X, y[f"cci_h{h}"], n_estimators=25, early_stopping_rounds=None) for h in (1, 2, 3)}
    ModelBundle(cci_models, clim, {
        "target": "Climate Comfort Index", "target_code": "cci", "prediction_type": "regression", "output_range": [0, 100],
        "algorithm": "XGBoost Regressor (test)", "forecast_horizon": "Next 3 days (t+1, t+2, t+3)", "horizons": [1, 2, 3],
        "features": features, "n_features": len(features), "feature_importance": {"cci": 0.5},
        "performance": {"test_2023_2025": {"t+1": {"mae": 5.0}}},
    }).save(root / "comfort_climate")

    y7 = y["whc_h7"].astype(int)
    whc_model = fit_xgb_classifier(X, y7, n_estimators=25, sample_weight=balanced_sample_weights(y7, power=0.5), early_stopping_rounds=None)
    ModelBundle({"whc_h7": whc_model}, clim, {
        "target": "Weather Hazard Category", "target_code": "whc", "prediction_type": "classification",
        "classes": {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk", 3: "Extreme Risk"},
        "algorithm": "XGBoost Classifier (test)", "forecast_horizon": "Exactly 7 days ahead (t+7)", "horizons": [7],
        "features": features, "n_features": len(features), "feature_importance": {"whi": 0.5},
        "performance": {"test_2023_2025": {"macro_f1": 0.4}},
    }).save(root / "weather_hazard")
    return root


@pytest.fixture(scope="session")
def client(models_dir, synthetic_daily):
    os.environ["MODELS_DIR"] = str(models_dir)
    from fastapi.testclient import TestClient

    from app import main

    # Replace the Open-Meteo call with a slice of the synthetic history.
    obs = synthetic_daily[observation_columns(synthetic_daily) + ["n_hours"]]

    def fake_window(end_date, lookback_days=90, location=None, today=None, session=None):
        end = pd.Timestamp(end_date)
        if end > obs.index.max():
            raise main.DataUnavailableError(f"no observation for {end.date()}")
        return obs.loc[end - pd.Timedelta(days=lookback_days): end].copy()

    main.fetch_daily_window = fake_window
    with TestClient(main.app) as c:
        yield c


@pytest.fixture(scope="session")
def valid_date(synthetic_daily):
    return synthetic_daily.index.max().date().isoformat()
