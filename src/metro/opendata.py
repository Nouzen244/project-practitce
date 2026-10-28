"""Загрузка открытых данных.

metro:     data.mos.ru, набор 62743 «Пассажиропоток по станциям Московского
           метрополитена» — входы/выходы по каждой станции за квартал
weather:   Open-Meteo archive (ERA5) — средняя температура и осадки по дням, Москва
calendar:  isdayoff.ru — производственный календарь РФ (с переносами)

Запуск:  python -m metro.opendata --start 2023-01-01 --end 2025-12-31 --out data/external
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd

DATA_MOS = "https://data.mos.ru/api/v2/odata"
METRO_DATASET_ID = 62743
OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"
MOSCOW = (55.7558, 37.6173)
ISDAYOFF = "https://isdayoff.ru/api/getdata"

QUARTERS = {"I квартал": 1, "II квартал": 2, "III квартал": 3, "IV квартал": 4}


def _request(url: str, body: dict | None = None, retries: int = 3) -> bytes:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json", "User-Agent": "metro-forecast/0.1"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


# ---------- data.mos.ru ----------
def fetch_metro_quarterly(dataset_id: int = METRO_DATASET_ID, page: int = 1000) -> pd.DataFrame:
    """Все строки набора: station, line, year, quarter, entries, exits."""
    meta = json.loads(_request(f"{DATA_MOS}/datasets/{dataset_id}"))
    catalog_id = meta["version"]["publicationCatalogId"]
    # epoch — дата текущего выпуска набора в формате, который ждёт API
    epoch = datetime.strptime(meta["actualData"]["releaseCreateDate"], "%d.%m.%Y %H:%M:%S")
    total = meta["release"]["cntObjects"]

    rows: list[dict] = []
    for offset in range(0, total, page):
        res = json.loads(_request(f"{DATA_MOS}/catalog/get", {
            "id": catalog_id, "epoch": epoch.strftime("%Y-%m-%d %H:%M:%S"), "timestamp": 1,
            "criteria": "", "limit": page, "offset": offset, "fetchGeodata": False}))
        if res.get("status") != 0:
            raise RuntimeError(f"data.mos.ru: {res.get('message')}")
        rows += res["response"]

    df = pd.DataFrame(rows)
    df = df[df["is_deleted"] == 0]
    df = pd.DataFrame({
        "station": df["NameOfStation"].str.strip(),
        "line": df["Line"].str.strip(),
        "year": df["Year"].astype(int),
        "quarter": df["Quarter"].map(QUARTERS).astype(int),
        "entries": df["IncomingPassengers"].astype(int),
        "exits": df["OutgoingPassengers"].astype(int),
    })
    # в наборе встречаются полные копии строк (≈10% на 2026-10) — убираем
    df = df.drop_duplicates()
    return df.sort_values(["line", "station", "year", "quarter"]).reset_index(drop=True)


# ---------- Open-Meteo ----------
def fetch_weather(start: str, end: str) -> pd.DataFrame:
    """Дневная погода по Москве: date, temp_c, precip_mm (схема как у synth)."""
    q = urllib.parse.urlencode({
        "latitude": MOSCOW[0], "longitude": MOSCOW[1], "start_date": start, "end_date": end,
        "daily": "temperature_2m_mean,precipitation_sum", "timezone": "Europe/Moscow"})
    d = json.loads(_request(f"{OPEN_METEO}?{q}"))["daily"]
    return pd.DataFrame({
        "date": pd.to_datetime(d["time"]).date,
        "temp_c": pd.Series(d["temperature_2m_mean"], dtype=float).round(1),
        "precip_mm": pd.Series(d["precipitation_sum"], dtype=float).round(1),
    })


# ---------- isdayoff.ru ----------
def fetch_calendar(start: str, end: str) -> pd.DataFrame:
    """date, is_holiday, is_dayoff.

    is_dayoff  — нерабочий день по производственному календарю (с выходными и переносами);
    is_holiday — нерабочий будний день (праздник или перенос), то, чего нет в признаке dow.
    """
    frames = []
    for year in range(int(start[:4]), int(end[:4]) + 1):
        flags = _request(f"{ISDAYOFF}?year={year}").decode().strip()
        dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        if len(flags) != len(dates):
            raise RuntimeError(f"isdayoff.ru: {len(flags)} флагов за {year}, ожидалось {len(dates)}")
        frames.append(pd.DataFrame({"date": dates, "is_dayoff": [c == "1" for c in flags]}))
    df = pd.concat(frames)
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    df["is_holiday"] = df["is_dayoff"] & (df["date"].dt.dayofweek < 5)
    df["date"] = df["date"].dt.date
    return df[["date", "is_holiday", "is_dayoff"]].reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--out", default="data/external")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, fetch in (("metro_quarterly", fetch_metro_quarterly),
                        ("weather", lambda: fetch_weather(a.start, a.end)),
                        ("calendar", lambda: fetch_calendar(a.start, a.end))):
        df = fetch()
        df.to_parquet(out / f"{name}.parquet", index=False)
        print(f"{name}: {len(df)} строк -> {out / name}.parquet")


if __name__ == "__main__":
    main()
