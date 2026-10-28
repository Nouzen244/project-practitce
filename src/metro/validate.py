"""Проверки качества данных перед загрузкой в БД.

error   — данные грузить нельзя (нарушена схема, дубли ключей, дыры в датах);
warning — грузить можно, но надо знать (неполные кварталы, нули, выбросы).

Запуск:  python -m metro.validate --data data/raw [--external data/external] --report reports/data_quality.json
Код выхода 1, если есть ошибки.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from metro.stations import HOURS, STATIONS

SCHEMA = {
    "ridership": ["date", "hour", "station_id", "event", "passengers"],
    "weather": ["date", "temp_c", "precip_mm"],
    "calendar": ["date", "is_holiday", "is_dayoff"],
}
KEYS = {"ridership": ["date", "hour", "station_id"], "weather": ["date"], "calendar": ["date"]}


@dataclass
class Issue:
    level: str
    table: str
    check: str
    detail: str


def check_raw(tables: dict[str, pd.DataFrame]) -> list[Issue]:
    issues: list[Issue] = []

    def err(t, c, d):
        issues.append(Issue("error", t, c, d))

    def warn(t, c, d):
        issues.append(Issue("warning", t, c, d))

    for name, cols in SCHEMA.items():
        df = tables[name]
        missing = set(cols) - set(df.columns)
        if missing:
            err(name, "schema", f"нет столбцов {sorted(missing)}")
            continue
        nulls = df[cols].isna().sum()
        for col, n in nulls[nulls > 0].items():
            err(name, "nulls", f"{col}: {n} пустых значений")
        dup = df.duplicated(KEYS[name]).sum()
        if dup:
            err(name, "duplicates", f"{dup} дублей по ключу {KEYS[name]}")

    if any(i.check == "schema" for i in issues):
        return issues

    # даты: непрерывность и совпадение диапазонов между таблицами
    r, w, c = tables["ridership"], tables["weather"], tables["calendar"]
    days = pd.to_datetime(pd.Series(sorted(r["date"].unique())))
    full = pd.date_range(days.min(), days.max(), freq="D")
    gaps = full.difference(days)
    if len(gaps):
        err("ridership", "date_gaps", f"{len(gaps)} пропущенных дней, первый {gaps[0].date()}")
    for name, df in (("weather", w), ("calendar", c)):
        lost = set(r["date"]) - set(df["date"])
        if lost:
            err(name, "coverage", f"нет {len(lost)} дней из ridership, первый {min(lost)}")

    # полнота сетки день × час × станция
    expected = len(full) * len(HOURS) * len(STATIONS)
    if len(r) != expected:
        err("ridership", "grid", f"{len(r)} строк, ожидалось {expected} (дни × часы × станции)")
    bad_h = sorted(set(r["hour"]) - set(HOURS))
    if bad_h:
        err("ridership", "hours", f"часы вне {HOURS[0]}–{HOURS[-1]}: {bad_h}")
    bad_s = sorted(set(r["station_id"]) - {s["id"] for s in STATIONS})
    if bad_s:
        err("ridership", "stations", f"неизвестные станции: {bad_s}")

    # значения
    if (r["passengers"] < 0).any():
        err("ridership", "range", f"{(r['passengers'] < 0).sum()} отрицательных значений")
    zero = (r["passengers"] == 0).mean()
    if zero > 0.01:
        warn("ridership", "zeros", f"{zero:.1%} нулевых часов")
    if not w["temp_c"].between(-45, 40).all():
        err("weather", "range", "температура вне −45…+40 °C")
    if not w["precip_mm"].between(0, 150).all():
        err("weather", "range", "осадки вне 0…150 мм")

    # выбросы: час станции > 4× медианы этой станции в этот час и тип дня
    # (тип дня — по производственному календарю: рабочая суббота считается рабочим днём)
    rr = r.merge(c[["date", "is_dayoff"]], on="date", how="left")
    med = rr.groupby(["station_id", "hour", "is_dayoff"])["passengers"].transform("median")
    out = rr[(rr["passengers"] > 4 * med) & (rr["event"] == 0)]
    if len(out):
        warn("ridership", "outliers", f"{len(out)} часов > 4× медианы без отметки события")
    return issues


def check_quarterly(q: pd.DataFrame, line: str = "Сокольническая линия") -> list[Issue]:
    """Полнота квартальных данных data.mos.ru по пилотной линии."""
    issues = []
    dup = q.duplicated(["station", "line", "year", "quarter"]).sum()
    if dup:
        issues.append(Issue("error", "metro_quarterly", "duplicates",
                            f"{dup} дублей по ключу станция/линия/год/квартал"))
    s = q[q["line"] == line]
    names = {st["name"] for st in STATIONS}
    absent = sorted(names - set(s["station"]))
    if absent:
        issues.append(Issue("warning", "metro_quarterly", "stations",
                            f"нет в data.mos.ru: {absent}"))
    extra = sorted(set(s["station"]) - names)
    if extra:
        issues.append(Issue("warning", "metro_quarterly", "stations",
                            f"есть в data.mos.ru, нет в справочнике: {extra}"))
    zeros = sorted(s.loc[s["entries"] == 0, "station"].unique())
    if zeros:
        issues.append(Issue("warning", "metro_quarterly", "zeros", f"нулевые входы: {zeros}"))
    per_q = s.groupby(["year", "quarter"])["station"].nunique()
    full = per_q.max()
    for (y, qq), n in per_q.items():
        if n < 0.8 * full:
            issues.append(Issue("warning", "metro_quarterly", "completeness",
                                f"{y} Q{qq}: {n} станций из {full}"))
    return issues


def load(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    d = Path(data_dir)
    out = {n: pd.read_parquet(d / f"{n}.parquet") for n in SCHEMA}
    for df in out.values():
        if "date" in df:
            df["date"] = pd.to_datetime(df["date"]).dt.date
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/raw")
    p.add_argument("--external")
    p.add_argument("--report", default="reports/data_quality.json")
    a = p.parse_args()
    issues = check_raw(load(a.data))
    if a.external:
        issues += check_quarterly(pd.read_parquet(Path(a.external) / "metro_quarterly.parquet"))
    n_err = sum(i.level == "error" for i in issues)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(
        {"errors": n_err, "warnings": len(issues) - n_err, "issues": [asdict(i) for i in issues]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    for i in issues:
        print(f"[{i.level}] {i.table}.{i.check}: {i.detail}")
    print(f"ошибок: {n_err}, предупреждений: {len(issues) - n_err} -> {a.report}")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
