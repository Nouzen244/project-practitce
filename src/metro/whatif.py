"""Сценарии «что-если» поверх базового прогноза.

Часть эффектов модель считает сама (событие у станции, погода — это её
признаки), часть задаётся правилами (закрытие станции, изменение интервала).
Коэффициенты правил — допущения проекта, их стоит уточнить на реальных данных.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from metro import model as M
from metro.stations import HOURS, neighbors

# Доля пассажиров закрытой станции, уходящих на соседние станции линии
REDIRECT_SHARE = 0.6
# Эластичность спроса по частоте движения: спрос ~ (старый интервал / новый) ** e
INTERVAL_ELASTICITY = 0.15


@dataclass
class Closure:
    station_id: int
    dates: list[date]
    hours: list[int] = field(default_factory=lambda: list(HOURS))


@dataclass
class Event:
    station_id: int
    date: date
    hours: list[int]


@dataclass
class WeatherOverride:
    date: date
    temp_c: float
    precip_mm: float


@dataclass
class IntervalChange:
    old_minutes: float
    new_minutes: float
    hours: list[int] = field(default_factory=lambda: list(HOURS))


@dataclass
class Scenario:
    closures: list[Closure] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    weather: list[WeatherOverride] = field(default_factory=list)
    interval: IntervalChange | None = None


def run(model, ridership_hist: pd.DataFrame, weather_hist: pd.DataFrame,
        scenario: Scenario, start: date | None = None) -> pd.DataFrame:
    """Возвращает date, hour, station_id, baseline, scenario, delta."""
    base = M.forecast(model, ridership_hist, weather_hist, start=start)
    days = sorted(base["date"].unique())

    # 1) события и погода — через признаки модели
    fut = M.future_frame(days[0], len(days))
    for ev in scenario.events:
        m = (fut["date"] == ev.date) & (fut["station_id"] == ev.station_id) & fut["hour"].isin(ev.hours)
        fut.loc[m, "event"] = 1
    wf = M.climate_weather(weather_hist, days)
    for w in scenario.weather:
        wf.loc[wf["date"] == w.date, ["temp_c", "precip_mm"]] = [w.temp_c, w.precip_mm]
    scen = M.forecast(model, ridership_hist, weather_hist, start=days[0], future=fut, weather_future=wf)
    scen = scen.rename(columns={"passengers_pred": "scenario"})
    df = base.rename(columns={"passengers_pred": "baseline"}).merge(
        scen, on=["date", "hour", "station_id"])
    df["scenario"] = df["scenario"].astype(float)

    # 2) изменение интервала движения — правило эластичности
    if scenario.interval and scenario.interval.new_minutes > 0:
        k = (scenario.interval.old_minutes / scenario.interval.new_minutes) ** INTERVAL_ELASTICITY
        df.loc[df["hour"].isin(scenario.interval.hours), "scenario"] *= k

    # 3) закрытие станций — обнуление и перераспределение на соседей
    key = ["date", "hour"]
    for c in scenario.closures:
        mask = (df["station_id"] == c.station_id) & df["date"].isin(c.dates) & df["hour"].isin(c.hours)
        lost = df.loc[mask, key + ["scenario"]].rename(columns={"scenario": "lost"})
        df.loc[mask, "scenario"] = 0
        nb = [n for n in neighbors(c.station_id)
              if n not in {x.station_id for x in scenario.closures}]
        if not nb:
            continue
        df = df.merge(lost, on=key, how="left")
        add = df["lost"].fillna(0) * REDIRECT_SHARE / len(nb)
        df.loc[df["station_id"].isin(nb), "scenario"] += add[df["station_id"].isin(nb)]
        df = df.drop(columns="lost")

    df["scenario"] = df["scenario"].round().astype(int)
    df["delta"] = df["scenario"] - df["baseline"]
    return df
