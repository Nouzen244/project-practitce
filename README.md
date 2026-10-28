# Прогноз пассажиропотока метро

Прогноз входов пассажиров по станциям одной линии (Сокольническая) на 7 дней вперёд с почасовой детализацией. Прогноз пересчитывается раз в день. Есть сценарии «что-если».

План на 8 недель и задачи команды: [docs/PLAN.md](docs/PLAN.md). Карточки для Kanban (импорт CSV в Jira/Trello): [docs/kanban.csv](docs/kanban.csv).

## Структура

```
src/metro/      данные и модель
  stations.py   справочник станций линии
  synth.py      генератор синтетических данных (схема как у реальных)
  features.py   признаки: календарь, погода, события, лаги 7/14 дней
  model.py      метрики, модели, прогноз на 7 дней
  train.py      CLI обучения
  whatif.py     сценарии «что-если»
src/app/        backend на FastAPI
  main.py       эндпоинты, метрики Prometheus, ежедневный пересчёт
  service.py    работа с моделью и БД, фильтры
  db.py         схема PostgreSQL
  etl.py        загрузка данных в БД
  schemas.py    схемы запросов
tests/          сквозные тесты API
monitoring/     Prometheus и Grafana
docs/           план и Kanban
```

## Быстрый старт (локально, SQLite)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
export PYTHONPATH=src                                   # Windows: set PYTHONPATH=src

python -m metro.synth --out data/raw                    # данные 2023–2025
python -m metro.train --data data/raw --out models      # обучение и метрики
python -m app.etl --data data/raw                       # загрузка в БД
uvicorn app.main:app --reload                           # http://localhost:8000/docs
pytest -q
```

## Docker (PostgreSQL + API + Prometheus + Grafana)

Сначала обучите модель (шаги synth и train выше), затем:

```bash
docker compose up -d --build
docker compose exec api python -m app.etl --data data/raw
curl -X POST localhost:8000/forecast/run
```

API: `localhost:8000/docs`, Prometheus: `localhost:9090`, Grafana: `localhost:3000` (admin/admin).

## API

| Метод | Путь | Что делает |
|---|---|---|
| GET | `/health` | статус, загружена ли модель, дата последних фактов |
| GET | `/stations` | станции линии с координатами |
| GET | `/history` | факт с фильтрами (по умолчанию последние 30 дней) |
| POST | `/forecast/run` | пересчитать прогноз на 7 дней |
| GET | `/forecast` | последний прогноз с фильтрами |
| POST | `/whatif` | базовый прогноз против сценария |
| GET | `/model/metrics` | метрики моделей на тестовом году |
| POST | `/model/reload` | перечитать файл модели |
| GET | `/metrics` | метрики для Prometheus |

**Фильтры** для `/history` и `/forecast`: `date_from`, `date_to`, `station_ids` (можно несколько), участок линии `segment_from` + `segment_to`, `hour_from`, `hour_to`. Параметр `agg`: `hour` (по часам), `day` (сумма за день по станции), `station` (сумма за период), `line_day` (вся линия по дням).

**Пример «что-если»:** закрыта «Лубянка» 5 января, интервал увеличен с 2 до 3 минут.

```json
POST /whatif
{
  "scenario": {
    "closures": [{"station_id": 9, "dates": ["2026-01-05"]}],
    "interval": {"old_minutes": 2, "new_minutes": 3}
  },
  "segment": [8, 10],
  "date_from": "2026-01-05",
  "date_to": "2026-01-05"
}
```

Доступные сценарии:

- **Закрытие станции**: поток станции обнуляется, 60% уходит на соседние станции.
- **Событие** у станции (матч, концерт): учитывается моделью как признак.
- **Погода** на конкретный день: температура и осадки подставляются в признаки модели.
- **Изменение интервала движения**: спрос меняется с эластичностью 0,15.

Коэффициенты закрытия и интервала — допущения проекта. Их надо уточнить на реальных данных.

## Модель и метрики

Обучение на 2023–2024 годах, тест на 2025 году (последний год, как договорились с заказчиком). Лаги не ближе 7 дней, поэтому все признаки известны на момент прогноза на неделю.

| Модель | MAPE | WAPE | Точность (100 − MAPE) |
|---|---|---|---|
| Сезонная наивная (неделю назад) | 13,2% | 11,0% | 86,8% |
| Ridge | 20,5% | 19,5% | 79,5% |
| LightGBM | 6,4% | 6,4% | **93,6%** |

Цель заказчика — выше 85%. **Важно:** это результат на синтетических данных. На реальных данных точность будет ниже, её надо пересчитать после замены источника.

MAPE считается по часам с потоком от 50 человек: в ранние часы проценты от малых чисел искажают метрику. MLP на PyTorch добавляется на неделе 5.

## Данные

Пока открытых данных недостаточно, используется синтетика. Она учитывает суточные профили по типам станций, выходные и праздники, лето, погоду, события у «Лужников», тренд и шум. Реальный источник подключается в ETL с той же схемой таблиц: `ridership(date, hour, station_id, event, passengers)`, `weather(date, temp_c, precip_mm)`, `calendar(date, is_holiday)`.
