"""Pydantic-схемы запросов и ответов."""
from __future__ import annotations

import datetime as dt
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from metro.stations import HOURS, STATION_BY_ID

Agg = Literal["hour", "day", "station", "line_day"]


def _check_station(v: int) -> int:
    if v not in STATION_BY_ID:
        raise ValueError(f"неизвестная станция {v}")
    return v


def _check_hours(v: list[int]) -> list[int]:
    bad = [h for h in v if h not in HOURS]
    if bad:
        raise ValueError(f"часы вне диапазона {HOURS[0]}–{HOURS[-1]}: {bad}")
    return v


class Station(BaseModel):
    id: int
    name: str
    line_id: int
    order: int
    lat: float
    lon: float
    kind: str


class ForecastRun(BaseModel):
    run_id: int
    start_date: date
    end_date: date
    rows: int
    model: str
    seconds: float


class ClosureIn(BaseModel):
    station_id: int
    dates: list[date]
    hours: list[int] = Field(default_factory=lambda: list(HOURS))
    @field_validator("station_id")
    @classmethod
    def v_station(cls, v):
        return _check_station(v)

    @field_validator("hours")
    @classmethod
    def v_hours(cls, v):
        return _check_hours(v)


class EventIn(BaseModel):
    station_id: int
    date: dt.date
    hours: list[int]
    @field_validator("station_id")
    @classmethod
    def v_station(cls, v):
        return _check_station(v)

    @field_validator("hours")
    @classmethod
    def v_hours(cls, v):
        return _check_hours(v)


class WeatherIn(BaseModel):
    date: dt.date
    temp_c: float = Field(ge=-50, le=50)
    precip_mm: float = Field(ge=0, le=200)


class IntervalIn(BaseModel):
    old_minutes: float = Field(gt=0, le=30)
    new_minutes: float = Field(gt=0, le=30)
    hours: list[int] = Field(default_factory=lambda: list(HOURS))
    @field_validator("hours")
    @classmethod
    def v_hours(cls, v):
        return _check_hours(v)


class ScenarioIn(BaseModel):
    """Сценарий «что-если» на ближайшие 7 дней."""
    closures: list[ClosureIn] = []
    events: list[EventIn] = []
    weather: list[WeatherIn] = []
    interval: IntervalIn | None = None


class WhatIfRequest(BaseModel):
    scenario: ScenarioIn
    station_ids: list[int] | None = None
    segment: tuple[int, int] | None = None
    date_from: date | None = None
    date_to: date | None = None
    hour_from: int | None = None
    hour_to: int | None = None
    agg: Agg = "day"


class WhatIfResponse(BaseModel):
    total_baseline: int
    total_scenario: int
    delta_pct: float
    rows: list[dict]
