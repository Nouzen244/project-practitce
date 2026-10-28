"""Признаки для модели.

Лаги берутся не ближе 7 дней (168 ч работы = 7 суток × тот же час), поэтому
при прогнозе на неделю вперёд все признаки известны на момент прогноза.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "station_id", "hour", "dow", "is_weekend", "is_holiday", "month",
    "doy_sin", "doy_cos", "temp_c", "precip_mm", "event",
    "lag_7d", "lag_14d", "mean_4w",
]
CATEGORICAL = ["station_id"]


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.to_datetime(df["date"])
    df["dow"] = d.dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["month"] = d.dt.month
    doy = d.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["is_holiday"] = df["is_holiday"].astype(int)
    return df


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Лаги того же часа и станции 7 и 14 дней назад и среднее за 4 недели."""
    df = df.sort_values(["station_id", "hour", "date"]).copy()
    g = df.groupby(["station_id", "hour"])["passengers"]
    df["lag_7d"] = g.shift(7)
    df["lag_14d"] = g.shift(14)
    df["mean_4w"] = (g.shift(7) + g.shift(14) + g.shift(21) + g.shift(28)) / 4
    return df


def build(ridership: pd.DataFrame, weather: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Склеивает таблицы и строит все признаки. Строки без лагов отбрасываются."""
    df = ridership.merge(weather, on="date", how="left").merge(calendar, on="date", how="left")
    df["is_holiday"] = df["is_holiday"].fillna(False)
    df = add_calendar(df)
    df = add_lags(df)
    return df.dropna(subset=["lag_7d", "lag_14d", "mean_4w"]).reset_index(drop=True)
