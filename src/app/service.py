"""Работа с моделью и БД: пересчёт прогноза, история, «что-если»."""
from __future__ import annotations

import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy import func, insert, select
from sqlalchemy.engine import Engine

from app import db
from app.logging_conf import log
from metro import model as M
from metro import whatif

logger = logging.getLogger("service")


class ModelNotLoaded(RuntimeError):
    pass


class ForecastService:
    def __init__(self, engine: Engine, model_path: str, metrics_path: str):
        self.engine = engine
        self.model_path = model_path
        self.metrics_path = metrics_path
        self.model = None
        self.reload_model()

    # ---------- модель ----------
    def reload_model(self) -> bool:
        p = Path(self.model_path)
        if p.exists():
            self.model = joblib.load(p)
            log(logger, "model loaded", path=str(p), model=self.model.name)
            return True
        log(logger, "model file not found", logging.WARNING, path=str(p))
        return False

    def metrics(self) -> dict | None:
        p = Path(self.metrics_path)
        return json.loads(p.read_text()) if p.exists() else None

    def _require_model(self):
        if self.model is None:
            raise ModelNotLoaded("Модель не загружена: обучите её (python -m metro.train)")
        return self.model

    # ---------- данные ----------
    def stations(self) -> list[dict]:
        with self.engine.connect() as c:
            rows = c.execute(select(db.stations).order_by(db.stations.c.order)).mappings().all()
        return [dict(r) for r in rows]

    def last_fact_date(self) -> date | None:
        with self.engine.connect() as c:
            return c.execute(select(func.max(db.ridership.c.date))).scalar()

    def _history(self, days: int = 35, until: date | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        until = until or self.last_fact_date()
        if until is None:
            raise ValueError("В БД нет фактических данных: запустите python -m app.etl")
        q = select(db.ridership).where(db.ridership.c.date > until - timedelta(days=days),
                                       db.ridership.c.date <= until)
        with self.engine.connect() as c:
            r = pd.read_sql(q, c)
            w = pd.read_sql(select(db.weather), c)
        for df in (r, w):
            df["date"] = pd.to_datetime(df["date"]).dt.date
        return r, w

    def history(self, f: Filters) -> pd.DataFrame:
        t = db.ridership
        q = select(t.c.date, t.c.hour, t.c.station_id, t.c.passengers)
        q = _apply_filters(q, t, f)
        with self.engine.connect() as c:
            df = pd.read_sql(q, c)
        return df

    # ---------- прогноз ----------
    def run_forecast(self, start: date | None = None) -> dict:
        model = self._require_model()
        t0 = time.perf_counter()
        hist_until = start - timedelta(days=1) if start else None
        r, w = self._history(until=hist_until)
        pred = M.forecast(model, r, w, start=start)
        start = min(pred["date"])
        with self.engine.begin() as c:
            run_id = c.execute(insert(db.forecast_runs).values(
                start_date=start, model=model.name, rows=len(pred))).inserted_primary_key[0]
            c.execute(insert(db.forecasts), [
                {"run_id": run_id, "date": row.date, "hour": int(row.hour),
                 "station_id": int(row.station_id), "passengers_pred": int(row.passengers_pred)}
                for row in pred.itertuples()])
        took = round(time.perf_counter() - t0, 2)
        log(logger, "forecast run", run_id=run_id, start=str(start), rows=len(pred), seconds=took)
        return {"run_id": run_id, "start_date": start, "end_date": max(pred["date"]),
                "rows": len(pred), "model": model.name, "seconds": took}

    def latest_run(self) -> dict | None:
        with self.engine.connect() as c:
            r = c.execute(select(db.forecast_runs).order_by(db.forecast_runs.c.id.desc()).limit(1)).mappings().first()
        return dict(r) if r else None

    def forecast(self, f: Filters) -> tuple[dict | None, pd.DataFrame]:
        run = self.latest_run()
        if run is None:
            return None, pd.DataFrame(columns=["date", "hour", "station_id", "passengers_pred"])
        t = db.forecasts
        q = select(t.c.date, t.c.hour, t.c.station_id, t.c.passengers_pred).where(t.c.run_id == run["id"])
        q = _apply_filters(q, t, f)
        with self.engine.connect() as c:
            df = pd.read_sql(q, c)
        return run, df

    def whatif(self, scenario: whatif.Scenario, f: Filters) -> pd.DataFrame:
        model = self._require_model()
        r, w = self._history()
        df = whatif.run(model, r, w, scenario)
        return _filter_df(df, f)


# ---------- фильтры ----------
class Filters:
    """Фильтры по дате, станциям/участку линии и времени."""

    def __init__(self, date_from=None, date_to=None, station_ids=None,
                 segment=None, hour_from=None, hour_to=None):
        self.date_from, self.date_to = date_from, date_to
        self.hour_from, self.hour_to = hour_from, hour_to
        ids = set(station_ids or [])
        if segment:  # участок линии: все станции между двумя по порядку
            a, b = sorted(segment)
            ids |= set(range(a, b + 1))
        self.station_ids = sorted(ids) or None


def _apply_filters(q, t, f: Filters):
    if f.date_from:
        q = q.where(t.c.date >= f.date_from)
    if f.date_to:
        q = q.where(t.c.date <= f.date_to)
    if f.station_ids:
        q = q.where(t.c.station_id.in_(f.station_ids))
    if f.hour_from is not None:
        q = q.where(t.c.hour >= f.hour_from)
    if f.hour_to is not None:
        q = q.where(t.c.hour <= f.hour_to)
    return q.order_by(t.c.date, t.c.hour, t.c.station_id)


def _filter_df(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    m = pd.Series(True, index=df.index)
    if f.date_from:
        m &= df["date"] >= f.date_from
    if f.date_to:
        m &= df["date"] <= f.date_to
    if f.station_ids:
        m &= df["station_id"].isin(f.station_ids)
    if f.hour_from is not None:
        m &= df["hour"] >= f.hour_from
    if f.hour_to is not None:
        m &= df["hour"] <= f.hour_to
    return df[m]


def aggregate(df: pd.DataFrame, value_cols: list[str], by: str) -> pd.DataFrame:
    """by: hour — как есть; day — сумма за день по станции; station — сумма за период."""
    if df.empty or by == "hour":
        return df
    keys = {"day": ["date", "station_id"], "station": ["station_id"], "line_day": ["date"]}[by]
    return df.groupby(keys, as_index=False)[value_cols].sum()
