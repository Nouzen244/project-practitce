"""Загрузка данных (parquet) в БД: проверка качества → полная перезаливка в одной транзакции.

python -m app.etl --data data/raw [--external data/external]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import delete, insert
from sqlalchemy.engine import Engine

from app import db
from app.config import settings
from app.logging_conf import log, setup
from metro import validate
from metro.stations import STATIONS

logger = logging.getLogger("etl")


class DataQualityError(RuntimeError):
    pass


def _insert_df(conn, table, df: pd.DataFrame, chunk: int = 20000) -> None:
    rows = df.to_dict(orient="records")
    for i in range(0, len(rows), chunk):
        conn.execute(insert(table), rows[i:i + chunk])


def load(engine: Engine, data_dir: str, external_dir: str | None = None) -> dict[str, int]:
    """Перезаливает справочник, факты, погоду и календарь; прогнозы сбрасываются.

    Всё в одной транзакции: при ошибке в БД остаются прежние данные.
    """
    tables = validate.load(data_dir)
    errors = [i for i in validate.check_raw(tables) if i.level == "error"]
    if errors:
        raise DataQualityError("; ".join(f"{i.table}.{i.check}: {i.detail}" for i in errors))
    quarterly = None
    if external_dir and (Path(external_dir) / "metro_quarterly.parquet").exists():
        quarterly = pd.read_parquet(Path(external_dir) / "metro_quarterly.parquet")

    db.init(engine)
    counts = {"stations": len(STATIONS)}
    with engine.begin() as conn:
        for t in (db.forecasts, db.forecast_runs, db.ridership, db.weather, db.calendar, db.stations):
            conn.execute(delete(t))
        conn.execute(insert(db.stations), [
            {k: s[k] for k in ("id", "name", "line_id", "order", "lat", "lon", "kind")} for s in STATIONS])
        for name in ("weather", "calendar", "ridership"):
            cols = [c.name for c in getattr(db, name).columns]
            _insert_df(conn, getattr(db, name), tables[name][cols])
            counts[name] = len(tables[name])
            log(logger, "loaded", table=name, rows=len(tables[name]))
        if quarterly is not None:
            conn.execute(delete(db.metro_quarterly))
            _insert_df(conn, db.metro_quarterly, quarterly)
            counts["metro_quarterly"] = len(quarterly)
            log(logger, "loaded", table="metro_quarterly", rows=len(quarterly))
    return counts


def main() -> None:
    setup(settings.log_level)
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/raw")
    p.add_argument("--external", default="data/external", help="квартальные данные data.mos.ru, если есть")
    a = p.parse_args()
    print(load(db.make_engine(settings.database_url), a.data, a.external))


if __name__ == "__main__":
    main()
