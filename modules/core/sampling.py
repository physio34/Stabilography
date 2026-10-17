"""Слой данных: дискретизация, накопление, фильтрация. Не зависит от UI."""
import collections
import threading
import time

import numpy as np


class DataBuffer:
    """Дискретизация, накопление, вычисление метрик.
    Не зависит от UI и конкретного типа устройства.
    """

    def __init__(self, cfg: dict, dev, xcode: int, ycode: int):
        self._cfg = cfg
        self._dev = dev
        self._xcode = xcode
        self._ycode = ycode

        self._samples: list[tuple[float, float, float, float]] = []
        self._lock = threading.Lock()
        self._start = time.time()
        self._last_sample = 0.0
        self._interval = 1.0 / cfg["rate"]
        self.running = True

        # Фильтрация
        self._median_win = max(int(cfg.get("median_window", 3)), 1)
        self._outlier_sigma = float(cfg.get("outlier_sigma", 5.0))
        self._buf_x: collections.deque = collections.deque(maxlen=self._median_win)
        self._buf_y: collections.deque = collections.deque(maxlen=self._median_win)
        self._running_mean_x = 0.0
        self._running_mean_y = 0.0
        self._running_var_x = 1.0
        self._running_var_y = 1.0
        self._n_seen = 0

    @property
    def elapsed(self) -> float:
        return time.time() - self._start

    @staticmethod
    def _median(buf: collections.deque) -> float:
        s = sorted(buf)
        n = len(s)
        if n == 0:
            return 0.0
        if n % 2 == 1:
            return s[n // 2]
        return (s[n // 2 - 1] + s[n // 2]) / 2.0

    def sample_once(self) -> bool:
        """Берёт один отсчёт, если прошёл нужный интервал.
        Применяет медианный фильтр и отсечение выбросов.
        """
        now = time.time()
        if now - self._last_sample < self._interval:
            return False
        self._last_sample = now

        xv = self._dev.get(self._xcode)
        yv = self._dev.get(self._ycode)
        if xv[0] is None or yv[0] is None:
            return False

        az = self._cfg["axis_zero"]
        us = self._cfg["unit_scale"]
        x_raw = (float(xv[0]) - az) * self._cfg["scale_x"] * us + self._cfg["offset_x"]
        y_raw = (float(yv[0]) - az) * self._cfg["scale_y"] * us + self._cfg["offset_y"]

        last_ts = max(xv[1], yv[1])
        contact = 1.0 if (now - last_ts) < self._cfg["contact_timeout"] else 0.0

        # Отсечение выбросов: если отклонение > outlier_sigma * sqrt(дисперсия) —
        # пропускаем точку (только если накопили достаточно данных)
        if self._n_seen >= 10 and self._outlier_sigma > 0:
            dx = abs(x_raw - self._running_mean_x)
            dy = abs(y_raw - self._running_mean_y)
            std_x = max(self._running_var_x, 1e-10) ** 0.5
            std_y = max(self._running_var_y, 1e-10) ** 0.5
            if dx > self._outlier_sigma * std_x or dy > self._outlier_sigma * std_y:
                return False

        # Медианный фильтр
        self._buf_x.append(x_raw)
        self._buf_y.append(y_raw)
        x = self._median(self._buf_x)
        y = self._median(self._buf_y)

        # Бегущее среднее и дисперсия (exponential moving average)
        alpha = 0.02
        self._n_seen += 1
        self._running_mean_x += alpha * (x - self._running_mean_x)
        self._running_mean_y += alpha * (y - self._running_mean_y)
        self._running_var_x += alpha * ((x - self._running_mean_x) ** 2 - self._running_var_x)
        self._running_var_y += alpha * ((y - self._running_mean_y) ** 2 - self._running_var_y)

        with self._lock:
            self._samples.append((now - self._start, x, y, contact))
        return True

    def sample_loop(self):
        """Фоновый поток выборки."""
        while self.running:
            self.sample_once()
            time.sleep(0.001)

    def snapshot(self) -> np.ndarray:
        """Потокобезопасная копия всех накопленных отсчётов (N×4)."""
        with self._lock:
            if not self._samples:
                return np.empty((0, 4))
            return np.array(self._samples)

    def to_array(self) -> np.ndarray:
        return self.snapshot()

    def stop(self):
        self.running = False
