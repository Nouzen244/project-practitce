"""Обучение, оценка и прогноз.

Обучение:  python -m metro.train --data data/raw --out models
Модели: сезонный наивный (значение неделю назад) → Ridge → LightGBM.
Валидация: обучаемся на всём до последнего года, тест — последний год
(так договорились с заказчиком). Лучшая по WAPE модель сохраняется.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from metro import features as F
from metro.stations import HOURS, STATIONS
from metro.synth import is_holiday

HORIZON_DAYS = 7


# ---------- метрики ----------
def mape(y, p, min_y: float = 50) -> float:
    """MAPE в %, без часов с очень малым потоком (< min_y), где проценты бессмысленны."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = y >= min_y
    return float(np.mean(np.abs(y[m] - p[m]) / y[m]) * 100)


def wape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.abs(y - p).sum() / y.sum() * 100)


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def report(y, p) -> dict:
    m = mape(y, p)
    return {"MAPE": round(m, 2), "WAPE": round(wape(y, p), 2), "MAE": round(mae(y, p), 1),
            "accuracy": round(100 - m, 2)}


# ---------- модели ----------
def make_ridge():
    num = [f for f in F.FEATURES if f not in F.CATEGORICAL + ["hour", "dow"]]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["station_id", "hour", "dow"]),
        ("num", StandardScaler(), num),
    ])
    return make_pipeline(pre, Ridge(alpha=1.0))


def make_lgbm():
    return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63,
                             min_child_samples=30, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.9, verbose=-1)


@dataclass
class Forecaster:
    """Обёртка над моделью: обучение на log1p, предсказание в пассажирах."""
    name: str
    est: object

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Forecaster:
        X = X.copy()
        if self.name == "lightgbm":
            X["station_id"] = X["station_id"].astype("category")
        self.est.fit(X, np.log1p(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X[F.FEATURES].copy()
        if self.name == "lightgbm":
            X["station_id"] = pd.Categorical(X["station_id"], categories=[s["id"] for s in STATIONS])
        return np.clip(np.expm1(self.est.predict(X)), 0, None)


def load_raw(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    d = Path(data_dir)
    out = {n: pd.read_parquet(d / f"{n}.parquet") for n in ("ridership", "weather", "calendar")}
    for df in out.values():
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return out


def train(data_dir: str, out_dir: str) -> dict:
    raw = load_raw(data_dir)
    df = F.build(raw["ridership"], raw["weather"], raw["calendar"])
    last_year = max(df["date"]).year
    tr = df[pd.to_datetime(df["date"]).dt.year < last_year]
    te = df[pd.to_datetime(df["date"]).dt.year == last_year]

    results = {"test_year": last_year, "train_rows": len(tr), "test_rows": len(te), "models": {}}
    results["models"]["seasonal_naive"] = report(te["passengers"], te["lag_7d"])

    fitted = {}
    for name, est in (("ridge", make_ridge()), ("lightgbm", make_lgbm())):
        f = Forecaster(name, est).fit(tr[F.FEATURES], tr["passengers"])
        results["models"][name] = report(te["passengers"], f.predict(te))
        fitted[name] = f
        print(name, results["models"][name])

    best = min(fitted, key=lambda n: results["models"][n]["WAPE"])
    results["best"] = best
    # финальная модель дообучается на всех данных
    final = Forecaster(best, make_lgbm() if best == "lightgbm" else make_ridge())
    final.fit(df[F.FEATURES], df["passengers"])

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    _log_mlflow(results)
    return results


def _log_mlflow(results: dict) -> None:
    """Если MLflow установлен — пишем метрики эксперимента."""
    try:
        import mlflow
    except ImportError:
        return
    with mlflow.start_run(run_name="baseline-comparison"):
        for model_name, m in results["models"].items():
            for k, v in m.items():
                mlflow.log_metric(f"{model_name}_{k}", v)
        mlflow.log_param("best", results["best"])


# ---------- прогноз ----------
def climate_weather(weather_hist: pd.DataFrame, days: list[date]) -> pd.DataFrame:
    """Погода на будущие дни = климатическая норма по дню года из истории."""
    w = weather_hist.copy()
    w["doy"] = pd.to_datetime(w["date"]).dt.dayofyear
    cols = ["temp_c", "precip_mm"]
    rows = []
    for d in days:
        doy = pd.Timestamp(d).dayofyear
        # окно ±7 дней по кругу года; если истории мало — берём последние 14 дней
        dist = np.minimum((w["doy"] - doy) % 366, (doy - w["doy"]) % 366)
        near = w[dist <= 7]
        if near.empty:
            near = w.sort_values("date").tail(14)
        rows.append({"date": d, **near[cols].mean().to_dict()})
    return pd.DataFrame(rows)


def future_frame(start: date, days: int = HORIZON_DAYS) -> pd.DataFrame:
    dates = [start + timedelta(days=i) for i in range(days)]
    idx = pd.MultiIndex.from_product([dates, HOURS, [s["id"] for s in STATIONS]],
                                     names=["date", "hour", "station_id"])
    df = idx.to_frame(index=False)
    df["event"] = 0
    df["passengers"] = np.nan
    return df


def forecast(model: Forecaster, ridership_hist: pd.DataFrame, weather_hist: pd.DataFrame,
             start: date | None = None, future: pd.DataFrame | None = None,
             weather_future: pd.DataFrame | None = None) -> pd.DataFrame:
    """Прогноз на 7 дней начиная с `start` (по умолчанию — день после конца истории).

    `future` можно передать заранее изменённым (события), `weather_future` — для
    сценариев с погодой. Возвращает date, hour, station_id, passengers_pred.
    """
    last = max(ridership_hist["date"])
    start = start or last + timedelta(days=1)
    if start > last + timedelta(days=1):
        raise ValueError("start должен быть не позже дня после последних фактических данных")
    fut = future if future is not None else future_frame(start)
    days = sorted(fut["date"].unique())
    wf = weather_future if weather_future is not None else climate_weather(weather_hist, days)
    cal = pd.DataFrame({"date": days, "is_holiday": [is_holiday(pd.Timestamp(d)) for d in days]})

    hist = ridership_hist[ridership_hist["date"] < start]
    hist = hist[hist["date"] >= start - timedelta(days=35)]
    hist = hist.merge(weather_hist, on="date", how="left")
    hist["is_holiday"] = [is_holiday(pd.Timestamp(d)) for d in hist["date"]]
    fut = fut.merge(wf, on="date", how="left").merge(cal, on="date", how="left")

    full = pd.concat([hist, fut], ignore_index=True)
    full = F.add_lags(F.add_calendar(full))
    pred_rows = full[full["passengers"].isna()].copy()
    pred_rows["passengers_pred"] = model.predict(pred_rows).round().astype(int)
    return pred_rows[["date", "hour", "station_id", "passengers_pred"]].sort_values(
        ["date", "hour", "station_id"]).reset_index(drop=True)
