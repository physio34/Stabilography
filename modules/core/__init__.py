"""Общий слой записи adStabilo: устройство, семплирование, метрики, CSV, дефолты."""
from .device import (
    Device,
    auto_detect_axes,
    discover_device,
    is_likely_stabilo_platform,
)
from .sampling import DataBuffer
from .metrics import compute_metrics, print_summary
from .io_csv import save_csv, build_csv_path
from .defaults import load_config

__all__ = [
    "Device",
    "auto_detect_axes",
    "discover_device",
    "is_likely_stabilo_platform",
    "DataBuffer",
    "compute_metrics",
    "print_summary",
    "save_csv",
    "build_csv_path",
    "load_config",
]
