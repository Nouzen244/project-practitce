"""Сквозные тесты: синтетика → обучение → ETL в SQLite → API."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client(tmp_path_factory, monkeypatch_session):
    tmp = tmp_path_factory.mktemp("metro")
    raw = tmp / "raw"
    from metro import model, synth

    raw.mkdir()
    for name, df in synth.generate("2024-10-01", "2025-01-31").items():
        df.to_parquet(raw / f"{name}.parquet", index=False)
    model.train(str(raw), str(tmp / "models"))

    monkeypatch_session.setenv("DATABASE_URL", f"sqlite:///{tmp / 'test.db'}")
    monkeypatch_session.setenv("MODEL_PATH", str(tmp / "models" / "model.joblib"))
    monkeypatch_session.setenv("METRICS_PATH", str(tmp / "models" / "metrics.json"))
    # env задаём до первого импорта app.* — настройки читаются при импорте
    import app.config
    import app.etl
    import app.main
    from app import db
    app.etl.load(db.make_engine(app.config.settings.database_url), str(raw))
    with TestClient(app.main.app) as c:
        yield c


@pytest.fixture(scope="session")
def monkeypatch_session():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["model_loaded"]
    assert r["last_fact_date"] == "2025-01-31"


def test_stations(client):
    s = client.get("/stations").json()
    assert len(s) == 26 and s[0]["name"] == "Бульвар Рокоссовского"


def test_history_filters(client):
    r = client.get("/history", params={"date_from": "2025-01-20", "date_to": "2025-01-21",
                                       "station_ids": [6], "hour_from": 8, "hour_to": 9}).json()
    assert len(r["rows"]) == 4
    assert {x["station_id"] for x in r["rows"]} == {6}


def test_forecast_before_run(client):
    # прогноз ещё не считали — честный 404
    assert client.get("/forecast").status_code in (200, 404)


def test_forecast_run_and_read(client):
    run = client.post("/forecast/run").json()
    assert run["start_date"] == "2025-02-01" and run["end_date"] == "2025-02-07"
    assert run["rows"] == 7 * 19 * 26
    r = client.get("/forecast", params={"segment_from": 6, "segment_to": 8, "agg": "day"}).json()
    assert {x["station_id"] for x in r["rows"]} == {6, 7, 8}
    assert len(r["rows"]) == 7 * 3
    assert all(x["passengers_pred"] > 0 for x in r["rows"])


def test_whatif_closure(client):
    d = "2025-02-03"
    body = {"scenario": {"closures": [{"station_id": 9, "dates": [d]}]},
            "date_from": d, "date_to": d, "segment": [8, 10], "agg": "day"}
    r = client.post("/whatif", json=body).json()
    rows = {x["station_id"]: x for x in r["rows"]}
    assert rows[9]["scenario"] == 0
    assert rows[8]["delta"] > 0 and rows[10]["delta"] > 0
    assert r["total_scenario"] < r["total_baseline"]  # часть пассажиров уходит с линии


def test_whatif_event_raises_flow(client):
    d = "2025-02-05"
    body = {"scenario": {"events": [{"station_id": 15, "date": d, "hours": [18, 19, 21]}]},
            "station_ids": [15], "date_from": d, "date_to": d, "agg": "day"}
    r = client.post("/whatif", json=body).json()
    assert r["delta_pct"] > 5


def test_whatif_validation(client):
    body = {"scenario": {"closures": [{"station_id": 99, "dates": ["2025-02-03"]}]}}
    assert client.post("/whatif", json=body).status_code == 422


def test_metrics_endpoints(client):
    assert "lightgbm" in client.get("/model/metrics").json()["models"]
    assert "forecast_runs_total" in client.get("/metrics").text
