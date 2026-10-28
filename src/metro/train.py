"""CLI обучения: python -m metro.train --data data/raw --out models

Отдельный модуль, чтобы класс модели сохранялся в pickle как
metro.model.Forecaster (а не __main__.Forecaster) и загружался в API.
"""
import argparse
import json

from metro.model import train


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/raw")
    p.add_argument("--out", default="models")
    a = p.parse_args()
    print(json.dumps(train(a.data, a.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
