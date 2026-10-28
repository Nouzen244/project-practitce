"""Настройки из переменных окружения."""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./metro.db")
    model_path: str = os.getenv("MODEL_PATH", "models/model.joblib")
    metrics_path: str = os.getenv("METRICS_PATH", "models/metrics.json")
    # Ежедневный пересчёт прогноза внутри процесса (в Kubernetes вместо него CronJob)
    scheduler_enabled: bool = os.getenv("SCHEDULER_ENABLED", "0") == "1"
    scheduler_hour_utc: int = int(os.getenv("SCHEDULER_HOUR_UTC", "1"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
