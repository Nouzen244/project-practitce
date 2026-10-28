"""Загрузка сырых данных (parquet) в БД.

python -m app.etl --data data/raw
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd
from sqlalchemy import delete, insert
from sqlalchemy.engine import Engine

from app import db
from app.config import settings
from app.logging_conf import log, setup
from metro.stations import STATIONS

logger = logging.getLogger("etl")


def load(engine: Engine, data_dir: str) -> dict[str, int]:
    db.init(engine)
    counts = {}
    with engine.begin() as conn:
        for t in (db.forecasts, db.forecast_runs, db.ridership, db.weather, db.calendar, db.stations):
            conn.execute(delete(t))
        conn.execute(insert(db.stations), [
            {k: s[k] for k in ("id", "name", "line_id", "order", "lat", "lon", "kind")} for s in STATIONS])
    counts["stations"] = len(STATIONS)

    for name in ("weather", "calendar", "ridership"):
        df = pd.read_parquet(f"{data_dir}/{name}.parquet")
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df.to_sql(name, engine, if_exists="append", index=False, chunksize=5000, method="multi")
        counts[name] = len(df)
        log(logger, "loaded", table=name, rows=len(df))
    return counts


def main() -> None:
    setup(settings.log_level)
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/raw")
    a = p.parse_args()
    print(load(db.make_engine(settings.database_url), a.data))


if __name__ == "__main__":
    main()
