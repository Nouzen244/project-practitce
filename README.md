# Прогноз пассажиропотока метро

Прогноз входов пассажиров по станциям одной линии (Сокольническая) на 7 дней вперёд с почасовой детализацией. Прогноз пересчитывается раз в день. Есть сценарии «что-если».

План на 8 недель и задачи команды: [docs/PLAN.md](docs/PLAN.md). Карточки для Kanban (импорт CSV в Jira/Trello): [docs/kanban.csv](docs/kanban.csv).

Отчёты по неделям: [неделя 1 — требования](docs/reports/week-01.md) · [неделя 2 — данные и ETL](docs/reports/week-02.md).

## Структура

```
src/metro/      данные и модель
  stations.py   справочник станций линии
  opendata.py   загрузка открытых данных: data.mos.ru, Open-Meteo, производственный календарь
  synth.py      генератор синтетических данных (схема как у реальных, калибровка по data.mos.ru)
  validate.py   проверки качества данных перед загрузкой
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
tests/          сквозные тесты API и тесты слоя данных
notebooks/      демонстрации по неделям (week2_demo.ipynb — данные)
monitoring/     Prometheus и Grafana
docs/           план, Kanban, требования (week1/), отчёты (reports/)
dvc.yaml        пайплайн данных: fetch → synth → validate
```

## Быстрый старт (локально, SQLite)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
export PYTHONPATH=src                                   # Windows: set PYTHONPATH=src

dvc repro                                               # открытые данные → синтетика → проверки
# без DVC: python -m metro.opendata && python -m metro.synth --external data/external
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

Данные 01.2023–10.2026. Обучение на 2023–2025 годах, тест на 2026 году (последний год, как договорились с заказчиком). Лаги не ближе 7 дней, поэтому все признаки известны на момент прогноза на неделю.

| Модель | MAPE | WAPE | Точность (100 − MAPE) |
|---|---|---|---|
| Сезонная наивная (неделю назад) | 14,3% | 12,0% | 85,7% |
| Ridge | 20,6% | 19,1% | 79,4% |
| LightGBM | 6,5% | 6,6% | **93,5%** |

Цель заказчика — выше 85%. **Важно:** это результат на синтетических данных. На реальных данных точность будет ниже, её надо пересчитать после замены источника.

MAPE считается по часам с потоком от 50 человек: в ранние часы проценты от малых чисел искажают метрику. MLP на PyTorch добавляется на неделе 5.

## Данные

Пока открытых данных недостаточно, используется синтетика. Она учитывает суточные профили по типам станций, выходные и праздники, лето, погоду, события у «Лужников», тренд и шум. Реальный источник подключается в ETL с той же схемой таблиц: `ridership(date, hour, station_id, event, passengers)`, `weather(date, temp_c, precip_mm)`, `calendar(date, is_holiday)`.
