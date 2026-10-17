"""CSV-экспорт стабилограмм. Единый формат заголовка: #Time,X,Y.

Колонка Time содержит абсолютное время в unix-формате (timestamp, секунды
с миллисекундной точностью): относительное время сэмпла + t_start.
Анализ нормализует её вычитанием первого значения, поэтому корректно
работают и старые файлы с относительным временем.
"""
import datetime
import os
from pathlib import Path

import numpy as np


def save_csv(arr: np.ndarray, path: str, meta: dict | None = None,
             t_start: float | None = None, t_end: float | None = None):
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        if meta:
            if meta.get("id"):
                f.write(f"#ID,{meta['id']}\n")
            if meta.get("gender"):
                f.write(f"#Gender,{meta['gender']}\n")
            if meta.get("age"):
                f.write(f"#Age,{meta['age']}\n")
        if t_start is not None:
            utc_start = datetime.datetime.fromtimestamp(t_start, tz=datetime.timezone.utc)
            f.write(f"#Start_UTC,{utc_start.strftime('%Y-%m-%dT%H:%M:%S.%f')}Z\n")
        if t_end is not None:
            utc_end = datetime.datetime.fromtimestamp(t_end, tz=datetime.timezone.utc)
            f.write(f"#End_UTC,{utc_end.strftime('%Y-%m-%dT%H:%M:%S.%f')}Z\n")
        f.write("#Time,X,Y\n")
        base = t_start if t_start is not None else 0.0
        for row in arr:
            f.write(f"{row[0] + base:.3f},{row[1]:.3f},{row[2]:.3f}\n")


def build_csv_path(base_path: str, meta: dict | None = None,
                   t_start: float | None = None) -> str:
    """Создаёт имя CSV-файла: base_YYYYMMDD_HHMMSS[_ID][_G][_Age].csv"""
    base, ext = os.path.splitext(base_path)
    parts = [base]
    if t_start is not None:
        utc = datetime.datetime.fromtimestamp(t_start, tz=datetime.timezone.utc)
        parts.append(utc.strftime("%Y%m%d_%H%M%S"))
    if meta:
        if meta.get("id"):
            parts.append(str(meta["id"]).replace(" ", "_"))
        if meta.get("gender"):
            parts.append(str(meta["gender"]))
        if meta.get("age"):
            parts.append(str(meta["age"]))
    return "_".join(parts) + ext


# Обратная совместимость со старым именем.
_build_csv_path = build_csv_path
