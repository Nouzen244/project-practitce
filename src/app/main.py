"""REST API прогноза пассажиропотока.

uvicorn app.main:app --reload       (из папки src, либо PYTHONPATH=src)
Документация: http://localhost:8000/docs
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime, timedelta
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from prometheus_client import Counter, Gauge
from prometheus_fastapi_instrumentator import Instrumentator

from app import db, schemas
from app.config import settings
from app.logging_conf import log, request_id, setup
from app.service import Filters, ForecastService, ModelNotLoaded, aggregate
from metro import whatif

logger = logging.getLogger("api")

FORECAST_RUNS = Counter("forecast_runs_total", "Пересчёты прогноза", ["status"])
FORECAST_AGE = Gauge("forecast_last_run_timestamp", "Время последнего пересчёта (unix)")
WHATIF_RUNS = Counter("whatif_requests_total", "Запросы «что-если»")


def get_service(request: Request) -> ForecastService:
    return request.app.state.service


Service = Annotated[ForecastService, Depends(get_service)]


def _do_run(svc: ForecastService, start: date | None = None) -> dict:
    try:
        res = svc.run_forecast(start)
        FORECAST_RUNS.labels("ok").inc()
        FORECAST_AGE.set(time.time())
        return res
    except Exception:
        FORECAST_RUNS.labels("error").inc()
        raise


async def _daily_scheduler(svc: ForecastService) -> None:
    """Пересчёт прогноза раз в сутки в settings.scheduler_hour_utc."""
    while True:
        now = datetime.now(UTC)
        nxt = now.replace(hour=settings.scheduler_hour_utc, minute=0, second=0, microsecond=0)
        if nxt <= now:
            nxt += timedelta(days=1)
        await asyncio.sleep((nxt - now).total_seconds())
        try:
            await asyncio.to_thread(_do_run, svc)
        except Exception:
            logger.exception("scheduled forecast failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup(settings.log_level)
    engine = db.make_engine(settings.database_url)
    db.init(engine)
    app.state.service = ForecastService(engine, settings.model_path, settings.metrics_path)
    task = asyncio.create_task(_daily_scheduler(app.state.service)) if settings.scheduler_enabled else None
    yield
    if task:
        task.cancel()


app = FastAPI(title="Metro Ridership Forecast API", version="0.1.0", lifespan=lifespan)
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


@app.middleware("http")
async def access_log(request: Request, call_next):
    rid = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
    request_id.set(rid)
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    log(logger, "request", method=request.method, path=request.url.path,
        status=response.status_code, ms=round((time.perf_counter() - t0) * 1000, 1))
    return response


def filters(
    date_from: date | None = None,
    date_to: date | None = None,
    station_ids: Annotated[list[int] | None, Query(description="id станций")] = None,
    segment_from: int | None = Query(None, description="участок линии: от станции (id)"),
    segment_to: int | None = Query(None, description="участок линии: до станции (id)"),
    hour_from: int | None = Query(None, ge=0, le=23),
    hour_to: int | None = Query(None, ge=0, le=23),
) -> Filters:
    if (segment_from is None) != (segment_to is None):
        raise HTTPException(422, "Для участка нужны оба параметра: segment_from и segment_to")
    seg = (segment_from, segment_to) if segment_from is not None else None
    return Filters(date_from, date_to, station_ids, seg, hour_from, hour_to)


Flt = Annotated[Filters, Depends(filters)]


@app.get("/health")
def health(svc: Service):
    return {"status": "ok", "model_loaded": svc.model is not None,
            "last_fact_date": svc.last_fact_date(), "latest_run": svc.latest_run()}


@app.get("/stations", response_model=list[schemas.Station])
def stations(svc: Service):
    return svc.stations()


@app.get("/history")
def history(svc: Service, f: Flt, agg: schemas.Agg = "hour"):
    """Фактический пассажиропоток. Без date_from — последние 30 дней."""
    if f.date_from is None:
        last = svc.last_fact_date()
        if last is None:
            raise HTTPException(404, "Нет фактических данных")
        f.date_from = last - timedelta(days=29)
    df = aggregate(svc.history(f), ["passengers"], agg)
    return {"rows": df.to_dict(orient="records")}


@app.post("/forecast/run", response_model=schemas.ForecastRun)
def forecast_run(svc: Service, start: date | None = None):
    """Пересчитать прогноз на 7 дней (вызывается раз в день по расписанию)."""
    try:
        return _do_run(svc, start)
    except ModelNotLoaded as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/forecast")
def forecast(svc: Service, f: Flt, agg: schemas.Agg = "hour"):
    """Последний сохранённый прогноз с фильтрами."""
    run, df = svc.forecast(f)
    if run is None:
        raise HTTPException(404, "Прогноз ещё не рассчитан: вызовите POST /forecast/run")
    df = aggregate(df, ["passengers_pred"], agg)
    return {"run": run, "rows": df.to_dict(orient="records")}


@app.post("/whatif", response_model=schemas.WhatIfResponse)
def what_if(svc: Service, req: schemas.WhatIfRequest):
    """Сравнение базового прогноза и сценария (закрытие станции, событие, погода, интервал)."""
    s = req.scenario
    last = svc.last_fact_date()
    if last is None:
        raise HTTPException(404, "Нет фактических данных")
    h_from, h_to = last + timedelta(days=1), last + timedelta(days=7)
    used = ([d for c in s.closures for d in c.dates] + [e.date for e in s.events]
            + [w.date for w in s.weather] + [d for d in (req.date_from, req.date_to) if d])
    outside = sorted({str(d) for d in used if not h_from <= d <= h_to})
    if outside:
        raise HTTPException(422, f"Даты вне горизонта прогноза {h_from}–{h_to}: {outside}")
    scen = whatif.Scenario(
        closures=[whatif.Closure(c.station_id, c.dates, c.hours) for c in s.closures],
        events=[whatif.Event(e.station_id, e.date, e.hours) for e in s.events],
        weather=[whatif.WeatherOverride(w.date, w.temp_c, w.precip_mm) for w in s.weather],
        interval=whatif.IntervalChange(s.interval.old_minutes, s.interval.new_minutes, s.interval.hours)
        if s.interval else None,
    )
    f = Filters(req.date_from, req.date_to, req.station_ids, req.segment, req.hour_from, req.hour_to)
    try:
        df = svc.whatif(scen, f)
    except ModelNotLoaded as e:
        raise HTTPException(503, str(e))
    WHATIF_RUNS.inc()
    base, sc = int(df["baseline"].sum()), int(df["scenario"].sum())
    out = aggregate(df, ["baseline", "scenario", "delta"], req.agg)
    return {"total_baseline": base, "total_scenario": sc,
            "delta_pct": round((sc - base) / base * 100, 2) if base else 0.0,
            "rows": out.to_dict(orient="records")}


@app.get("/model/metrics")
def model_metrics(svc: Service):
    m = svc.metrics()
    if m is None:
        raise HTTPException(404, "Метрики не найдены")
    return m


@app.post("/model/reload")
def model_reload(svc: Service):
    return {"reloaded": svc.reload_model()}
