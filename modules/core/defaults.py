"""Единые дефолты записи: формула шкалы (raw - axis_zero) * scale * unit_scale + offset."""
import json

_DEFAULTS: dict = {
    "xcode": 0, "ycode": 1, "rate": 50.0,
    "contact_timeout": 0.5, "calibrate": 2.0,
    "scale_x": 1.0, "scale_y": 1.0,
    "offset_x": 0.0, "offset_y": 0.0,
    "axis_zero": 128,
    "unit_scale": 0.1,
    "median_window": 3,
    "outlier_sigma": 5.0,
    "device": None, "max_x": None, "max_y": None,
}


def load_config(path: str | None) -> dict:
    cfg = dict(_DEFAULTS)
    if path:
        with open(path) as f:
            data = json.load(f)
        # Миграция старых axes.json (без axis_zero/unit_scale):
        # раньше шкала была x = raw * scale_x + offset_x.
        if "axis_zero" not in data and "unit_scale" not in data:
            data = dict(data)
            data.setdefault("axis_zero", 0)
            data.setdefault("unit_scale", 1.0)
        cfg.update(data)
    cfg["rate"] = max(float(cfg.get("rate") or 50.0), 0.1)
    cfg["contact_timeout"] = float(cfg["contact_timeout"])
    return cfg
