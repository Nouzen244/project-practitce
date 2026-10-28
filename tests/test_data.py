"""Тесты слоя данных (нед. 2): загрузчики открытых данных, калибровка, проверки, ETL."""
import json

import pandas as pd
import pytest
import sqlalchemy as sa

from app import db, etl
from metro import opendata, synth, validate
from metro.stations import STATIONS


@pytest.fixture(scope="module")
def small():
    return synth.generate("2025-01-01", "2025-02-28")


def test_synth_schema_passes_checks(small):
    assert [i for i in validate.check_raw(small) if i.level == "error"] == []
    assert set(small["calendar"].columns) == {"date", "is_holiday", "is_dayoff"}


@pytest.mark.parametrize("break_it, check", [
    (lambda t: t["ridership"].drop(index=0), "grid"),
    (lambda t: pd.concat([t["ridership"], t["ridership"].head(1)]), "duplicates"),
    (lambda t: t["ridership"].assign(passengers=-1), "range"),
    (lambda t: t["ridership"][t["ridership"]["date"] != t["ridership"]["date"].iloc[500]], "date_gaps"),
])
def test_checks_catch_errors(small, break_it, check):
    broken = {**small, "ridership": break_it(small)}
    assert check in {i.check for i in validate.check_raw(broken) if i.level == "error"}


def test_working_saturday_is_not_outlier():
    # 1 ноября 2025 — рабочая суббота: утренний пик будня не должен считаться выбросом
    dates = pd.date_range("2025-10-01", "2025-11-30")
    cal = pd.DataFrame({"date": dates.date, "is_holiday": False,
                        "is_dayoff": (dates.dayofweek >= 5) & (dates != "2025-11-01")})
    t = synth.generate("2025-10-01", "2025-11-30", calendar=cal)
    assert "outliers" not in {i.check for i in validate.check_raw(t)}


def test_calibration_recovers_station_volume():
    start, end = "2024-01-01", "2024-12-31"
    t = synth.generate(start, end)
    r = t["ridership"].assign(date=pd.to_datetime(t["ridership"]["date"]))
    by_q = r.groupby(["station_id", r["date"].dt.year.rename("year"),
                      r["date"].dt.quarter.rename("quarter")])["passengers"].sum().reset_index()
    names = {s["id"]: s["name"] for s in STATIONS}
    q = by_q.assign(station=by_q["station_id"].map(names), line="Сокольническая линия",
                    entries=by_q["passengers"], exits=by_q["passengers"])
    cal = t["calendar"]
    bases = synth.calibrate_bases(q, cal, start, end)
    for s in STATIONS:
        assert bases[s["id"]] == pytest.approx(s["base"], rel=0.05)


def test_fetch_metro_quarterly_parses_and_dedups(monkeypatch):
    meta = {"version": {"publicationCatalogId": 1},
            "actualData": {"releaseCreateDate": "23.10.2026 00:00:38"}, "release": {"cntObjects": 3}}
    row = {"global_id": 1, "is_deleted": 0, "NameOfStation": "Лубянка ", "Line": "Сокольническая линия",
           "Year": 2025, "Quarter": "III квартал", "IncomingPassengers": 10, "OutgoingPassengers": 9}
    rows = [row, row, {**row, "is_deleted": 1, "Year": 2024}]

    def fake(url, body=None, retries=3):
        if body is None:
            return json.dumps(meta).encode()
        assert body["epoch"] == "2026-10-23 00:00:38"
        return json.dumps({"status": 0, "response": rows[body["offset"]:body["offset"] + body["limit"]]}).encode()

    monkeypatch.setattr(opendata, "_request", fake)
    df = opendata.fetch_metro_quarterly(page=2)
    assert df.to_dict(orient="records") == [
        {"station": "Лубянка", "line": "Сокольническая линия", "year": 2025, "quarter": 3,
         "entries": 10, "exits": 9}]


def test_fetch_calendar_marks_weekday_holidays(monkeypatch):
    flags = "".join("1" if d.dayofweek >= 5 or d.month == 1 and d.day <= 8 else "0"
                    for d in pd.date_range("2025-01-01", "2025-12-31"))
    monkeypatch.setattr(opendata, "_request", lambda url, body=None, retries=3: flags.encode())
    c = opendata.fetch_calendar("2025-01-01", "2025-01-31").set_index("date")
    assert c.loc[pd.Timestamp("2025-01-03").date(), "is_holiday"]       # пятница, каникулы
    assert not c.loc[pd.Timestamp("2025-01-04").date(), "is_holiday"]   # суббота — выходной, не праздник
    assert c.loc[pd.Timestamp("2025-01-04").date(), "is_dayoff"]


def test_etl_rejects_bad_data_and_keeps_old(tmp_path, small):
    eng = db.make_engine(f"sqlite:///{tmp_path / 'etl.db'}")
    good, bad = tmp_path / "good", tmp_path / "bad"
    for d, tables in ((good, small), (bad, {**small, "ridership": small["ridership"].iloc[1:]})):
        d.mkdir()
        for name, df in tables.items():
            df.to_parquet(d / f"{name}.parquet", index=False)

    assert etl.load(eng, str(good))["ridership"] == len(small["ridership"])
    with pytest.raises(etl.DataQualityError, match="grid"):
        etl.load(eng, str(bad))
    with eng.connect() as c:
        assert c.execute(sa.text("select count(*) from ridership")).scalar() == len(small["ridership"])
