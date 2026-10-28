"""Генератор синтетических данных о пассажиропотоке.

Нужен, пока нет достаточного объёма открытых данных (риск из задания).
Схема совпадает с той, что ETL пишет из реальных источников:

weather:   date, temp_c, precip_mm
calendar:  date, is_holiday
ridership: date, hour, station_id, event, passengers   (входы за час)

Запуск:  python -m metro.synth --start 2023-01-01 --end 2025-12-31 --out data/raw
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from metro.stations import HOURS, STATIONS

# Российские нерабочие праздники (месяц, день)
HOLIDAYS = {(1, d) for d in range(1, 9)} | {(2, 23), (3, 8), (5, 1), (5, 9), (6, 12), (11, 4)}

# Доля суточного потока по часам 5..23 для разных типов станций
_PROFILES = {
    "residential": [1, 5, 11, 13, 8, 5, 4, 4, 4, 4, 5, 6, 7, 7, 5, 4, 3, 2, 1],
    "center":      [0.5, 1, 4, 7, 6, 5, 5, 6, 6, 6, 7, 11, 13, 9, 6, 4, 3, 2, 1],
    "mixed":       [1, 3, 8, 10, 7, 5, 5, 5, 5, 5, 6, 8, 9, 7, 5, 4, 3, 2, 1],
    "hub":         [2, 4, 8, 10, 8, 6, 6, 6, 6, 6, 7, 9, 9, 7, 6, 5, 4, 3, 2],
    "event":       [1, 3, 7, 8, 6, 5, 5, 5, 5, 6, 7, 9, 9, 8, 7, 6, 4, 3, 2],
}
_WEEKEND_PROFILE = [0.5, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 7, 7, 6, 5, 4, 3, 2]


def is_holiday(d: pd.Timestamp) -> bool:
    return (d.month, d.day) in HOLIDAYS


def make_weather(dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    doy = dates.dayofyear.values
    temp = 6 - 14 * np.cos(2 * np.pi * (doy - 15) / 365) + rng.normal(0, 4, len(dates))
    rain_day = rng.random(len(dates)) < 0.35
    precip = np.where(rain_day, rng.gamma(1.3, 3.5, len(dates)), 0.0)
    return pd.DataFrame({"date": dates.date, "temp_c": temp.round(1), "precip_mm": precip.round(1)})


def make_events(dates: pd.DatetimeIndex, rng: np.random.Generator) -> set[tuple]:
    """Матчи/концерты у Лужников: ~3 вечерних события в месяц, станции 15–16."""
    events = set()
    for d in dates:
        if rng.random() < 0.1:
            for sid in (15, 16):
                for h in (17, 18, 19, 21, 22):
                    events.add((d.date(), h, sid))
    return events


def generate(start: str, end: str, seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    weather = make_weather(dates, rng)
    calendar = pd.DataFrame({"date": dates.date, "is_holiday": [is_holiday(d) for d in dates]})
    events = make_events(dates, rng)

    w = weather.set_index("date")
    t0 = dates[0]
    rows = []
    for d in dates:
        day = d.date()
        weekend = d.dayofweek >= 5 or is_holiday(d)
        # уровень дня: выходные/праздники, лето, тренд, погода
        level = 1.0
        if weekend:
            level *= 0.55 if d.dayofweek == 6 or is_holiday(d) else 0.65
        if d.month in (7, 8):
            level *= 0.85
        if d.month == 1 and d.day <= 8:
            level *= 0.8
        level *= 1 + 0.02 * (d - t0).days / 365
        temp, precip = w.loc[day, "temp_c"], w.loc[day, "precip_mm"]
        if precip > 8:
            level *= 1.05
        if temp < -15:
            level *= 1.04
        day_noise = rng.normal(1, 0.03)

        for st in STATIONS:
            prof = np.array(_WEEKEND_PROFILE if weekend else _PROFILES[st["kind"]], dtype=float)
            prof /= prof.sum()
            st_noise = rng.lognormal(0, 0.06, len(HOURS))
            vals = st["base"] * level * day_noise * prof * st_noise
            for i, h in enumerate(HOURS):
                ev = (day, h, st["id"]) in events
                v = vals[i] * (2.5 if ev else 1.0)
                rows.append((day, h, st["id"], int(ev), round(v)))

    ridership = pd.DataFrame(rows, columns=["date", "hour", "station_id", "event", "passengers"])
    return {"weather": weather, "calendar": calendar, "ridership": ridership}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--out", default="data/raw")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in generate(a.start, a.end, a.seed).items():
        df.to_parquet(out / f"{name}.parquet", index=False)
        print(f"{name}: {len(df)} строк -> {out / name}.parquet")


if __name__ == "__main__":
    main()
