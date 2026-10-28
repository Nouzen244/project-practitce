"""Схема БД (PostgreSQL в проде, SQLite локально и в тестах)."""
from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    func,
)
from sqlalchemy.engine import Engine

metadata = MetaData()

stations = Table(
    "stations", metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String, nullable=False),
    Column("line_id", Integer, nullable=False),
    Column("order", Integer, nullable=False),
    Column("lat", Float),
    Column("lon", Float),
    Column("kind", String),
)

ridership = Table(
    "ridership", metadata,
    Column("date", Date, primary_key=True),
    Column("hour", Integer, primary_key=True),
    Column("station_id", Integer, ForeignKey("stations.id"), primary_key=True),
    Column("event", Integer, default=0),
    Column("passengers", Integer, nullable=False),
)

weather = Table(
    "weather", metadata,
    Column("date", Date, primary_key=True),
    Column("temp_c", Float),
    Column("precip_mm", Float),
)

calendar = Table(
    "calendar", metadata,
    Column("date", Date, primary_key=True),
    Column("is_holiday", Boolean, nullable=False),
)

forecast_runs = Table(
    "forecast_runs", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime, server_default=func.now()),
    Column("start_date", Date, nullable=False),
    Column("model", String),
    Column("rows", Integer),
)

forecasts = Table(
    "forecasts", metadata,
    Column("run_id", Integer, ForeignKey("forecast_runs.id"), primary_key=True),
    Column("date", Date, primary_key=True),
    Column("hour", Integer, primary_key=True),
    Column("station_id", Integer, ForeignKey("stations.id"), primary_key=True),
    Column("passengers_pred", Integer, nullable=False),
)


def make_engine(url: str) -> Engine:
    kw = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {"pool_pre_ping": True}
    return create_engine(url, **kw)


def init(engine: Engine) -> None:
    metadata.create_all(engine)
