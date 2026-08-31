#!/usr/bin/env python3
"""Стабилограмма MERA с Dear PyGui.

Архитектура (Loose Coupling):
  Device        — чтение evdev, валидация, кольцевой буфер
  DataBuffer    — накопление, дискретизация, вычисление метрик
  XInputGuard   — изоляция устройства ввода от X11 (контекст-менеджер)
  StabiloUI     — только Dear PyGui / визуализация
  main()        — сборка и запуск
"""

import argparse
import collections
import contextlib
import fcntl
import json
import os
import select
import struct
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd
import dearpygui.dearpygui as dpg

# ── evdev constants ────────────────────────────────────────────────────────────
EV_ABS = 3
EVENT_FMT = "llHHi"
EVENT_SIZE = struct.calcsize(EVENT_FMT)
ABSINFO_FMT = "iiiiii"
ABSINFO_SIZE = struct.calcsize(ABSINFO_FMT)

# ── цвета UI ───────────────────────────────────────────────────────────────────
COLOR_TRAJ_LINE   = [128, 128, 128, 102]
COLOR_TRAJ_PTS    = [51,  204,  77, 178]
COLOR_CURSOR      = [255, 220,  50, 230]
COLOR_X_LINE      = [51,  102, 255, 255]
COLOR_Y_LINE      = [255,  77,  77, 255]
COLOR_CONTACT_ON  = [50,  220,  80, 255]
COLOR_CONTACT_OFF = [220,  80,  80, 255]
COLOR_TITLE       = [102, 179, 255, 255]

WINDOW_SEC = 10.0   # скользящее окно временного ряда

# Пути к системным шрифтам с кириллицей (первый найденный будет использован)
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
]


# ══════════════════════════════════════════════════════════════════════════════
# Слой устройства
# ══════════════════════════════════════════════════════════════════════════════

def _eviocgabs(fd: int, code: int) -> tuple:
    """Возвращает (value, min, max, fuzz, flat, resolution) оси evdev."""
    request = (2 << 30) | (ABSINFO_SIZE << 16) | (ord("E") << 8) | (0x40 + code)
    buf = bytearray(ABSINFO_SIZE)
    fcntl.ioctl(fd, request, buf)
    return struct.unpack(ABSINFO_FMT, buf)


def _read_sysfs_text(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def discover_device() -> str | None:
    """Ищет первое устройство с 'MERA' в имени в /sys/class/input."""
    base = "/sys/class/input"
    try:
        names = os.listdir(base)
    except OSError:
        return None
    for name in sorted(names):
        if not name.startswith("event"):
            continue
        devname = _read_sysfs_text(os.path.join(base, name, "device", "name"))
        if devname and "MERA" in devname.upper():
            return os.path.join("/dev/input", name)
    return None


def _get_abs_capabilities(dev_path: str) -> list[int] | None:
    """Возвращает список кодов ABS-осей устройства."""
    sys_path = dev_path.replace("/dev/input/", "/sys/class/input/")
    cap_hex = _read_sysfs_text(os.path.join(sys_path, "device", "capabilities", "abs"))
    if cap_hex is None:
        return None
    try:
        cap_val = int(cap_hex, 16)
        return [i for i in range(64) if cap_val & (1 << i)]
    except ValueError:
        return None


def is_likely_stabilo_platform(dev_path: str) -> bool:
    """Эвристика: стабилограф имеет ≤5 осей без мультитач-кодов 47–63."""
    caps = _get_abs_capabilities(dev_path)
    if caps is None:
        return True
    if len(caps) > 5:
        return False
    return not any(c in range(47, 64) for c in caps)


class Device:
    """Низкоуровневое чтение evdev + валидация значений."""

    def __init__(self, path: str, max_x: int | None = None, max_y: int | None = None):
        self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        self._poll = select.poll()
        self._poll.register(self.fd, select.POLLIN)

        self._raw: collections.deque = collections.deque(maxlen=400_000)
        self._latest: dict = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

        self.max_x = max_x
        self.max_y = max_y
        self.valid_codes: set[int] = set(_get_abs_capabilities(path) or [])
        self.axis_info: dict[int, tuple[int, int]] = {}

        for code in list(self.valid_codes):
            try:
                _, mn, mx, _, _, _ = _eviocgabs(self.fd, code)
                self.axis_info[code] = (mn, mx)
            except (IOError, OSError, ValueError):
                pass

    def start(self):
        self._thread.start()

    def stop_now(self):
        self._stop.set()

    def close(self):
        with contextlib.suppress(OSError):
            os.close(self.fd)

    def get(self, code: int) -> tuple[int | None, float]:
        """Возвращает (last_value, timestamp)."""
        with self._lock:
            return self._latest.get(code, (None, 0.0))

    def recent_codes(self, since: float) -> dict[int, list[int]]:
        """События по кодам, начиная с момента since."""
        codes: dict[int, list[int]] = {}
        with self._lock:
            items = list(self._raw)
        for t, code, val in items:
            if t >= since:
                codes.setdefault(code, []).append(val)
        return codes

    def _validate(self, code: int, val: int) -> bool:
        if self.valid_codes and code not in self.valid_codes:
            return False
        if code in self.axis_info:
            mn, mx = self.axis_info[code]
            if not (mn <= val <= mx):
                return False
        if self.max_x is not None and code == 0 and val > self.max_x:
            return False
        if self.max_y is not None and code == 1 and val > self.max_y:
            return False
        return True

    def _run(self):
        while not self._stop.is_set():
            if not self._poll.poll(100):
                continue
            try:
                data = os.read(self.fd, EVENT_SIZE * 64)
            except OSError:
                continue
            now = time.time()
            for off in range(0, len(data) - EVENT_SIZE + 1, EVENT_SIZE):
                _, _, typ, code, val = struct.unpack(
                    EVENT_FMT, data[off: off + EVENT_SIZE]
                )
                if typ == EV_ABS and self._validate(code, val):
                    with self._lock:
                        self._latest[code] = (val, now)
                        self._raw.append((now, code, val))


def auto_detect_axes(dev: Device, window_sec: float) -> tuple[int, int]:
    """Определяет коды X/Y по дисперсии сигнала за window_sec секунд."""
    since = time.time()
    time.sleep(window_sec)
    codes = dev.recent_codes(since)
    if not codes:
        print("Предупреждение: нет данных за время калибровки, оси 0 и 1.", file=sys.stderr)
        return 0, 1

    def _range(code: int) -> int:
        mn, mx = dev.axis_info.get(code, (0, 0))
        return mx - mn

    ranked = sorted(
        codes,
        key=lambda c: (_range(c), float(np.var(codes[c])) if len(codes[c]) > 1 else 0.0),
        reverse=True,
    )
    if len(ranked) >= 2:
        return ranked[0], ranked[1]
    if len(ranked) == 1:
        return ranked[0], ranked[0]
    return 0, 1


# ══════════════════════════════════════════════════════════════════════════════
# Слой данных
# ══════════════════════════════════════════════════════════════════════════════

class DataBuffer:
    """Дискретизация, накопление, вычисление метрик.
    Не зависит от UI и конкретного типа устройства.
    """

    def __init__(self, cfg: dict, dev: Device, xcode: int, ycode: int):
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

    @property
    def elapsed(self) -> float:
        return time.time() - self._start

    def sample_once(self) -> bool:
        """Берёт один отсчёт, если прошёл нужный интервал.
        Возвращает True, если отсчёт добавлен.
        """
        now = time.time()
        if now - self._last_sample < self._interval:
            return False
        self._last_sample = now

        xv = self._dev.get(self._xcode)
        yv = self._dev.get(self._ycode)
        if xv[0] is None or yv[0] is None:
            return False

        x = float(xv[0]) * self._cfg["scale_x"] + self._cfg["offset_x"]
        y = float(yv[0]) * self._cfg["scale_y"] + self._cfg["offset_y"]
        last_ts = max(xv[1], yv[1])
        contact = 1.0 if (now - last_ts) < self._cfg["contact_timeout"] else 0.0

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


def compute_metrics(arr: np.ndarray) -> dict | None:
    """Вычисляет стабилографические показатели из массива N×4 [t, x, y, contact]."""
    if arr.size == 0:
        return None
    t, x, y, c = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    sel = c > 0.5
    used_all = not sel.any()
    if used_all:
        sel = np.ones(len(c), dtype=bool)
    xx, yy, tt = x[sel], y[sel], t[sel]
    if len(xx) < 2:
        return dict(n=int(len(xx)), T=0.0,
                    xm=float(xx.mean()) if len(xx) else 0.0,
                    ym=float(yy.mean()) if len(yy) else 0.0,
                    sx=0.0, sy=0.0, maxdev_x=0.0, maxdev_y=0.0,
                    range_x=0.0, range_y=0.0, L=0.0, vel=0.0, rms=0.0,
                    no_contact=used_all)
    xm, ym = xx.mean(), yy.mean()
    dx, dy = np.diff(xx), np.diff(yy)
    L = float(np.sqrt(dx**2 + dy**2).sum())
    T = float(tt[-1] - tt[0])
    return dict(
        n=int(len(xx)), T=T,
        xm=float(xm), ym=float(ym),
        sx=float(xx.std()), sy=float(yy.std()),
        maxdev_x=float(np.abs(xx - xm).max()),
        maxdev_y=float(np.abs(yy - ym).max()),
        range_x=float(xx.max() - xx.min()),
        range_y=float(yy.max() - yy.min()),
        L=L, vel=L / T if T > 0 else 0.0,
        rms=float(np.sqrt(((xx - xm)**2 + (yy - ym)**2).mean())),
        no_contact=used_all,
    )


def save_csv(arr: np.ndarray, path: str):
    if not path:
        return
    df = pd.DataFrame(arr[:, :3], columns=["Time", "X", "Y"])
    df.to_csv(path, index=False, float_format="%.6f")


def print_summary(m: dict | None):
    if not m:
        print("Нет данных для анализа.")
        return
    print("=== Стабилографические показатели ===")
    rows = [
        ("Число отсчётов (контакт)", f"{m['n']}"),
        ("Длительность записи, с",   f"{m['T']:.2f}"),
        ("Среднее положение X / Y",  f"{m['xm']:.2f} / {m['ym']:.2f}"),
        ("σ по X / по Y",            f"{m['sx']:.2f} / {m['sy']:.2f}"),
        ("max откл. X / Y",          f"{m['maxdev_x']:.2f} / {m['maxdev_y']:.2f}"),
        ("Размах X / Y",             f"{m['range_x']:.2f} / {m['range_y']:.2f}"),
        ("Длина тракта L",           f"{m['L']:.2f}"),
        ("Средняя скорость",         f"{m['vel']:.2f} ед/с"),
        ("RMS отклонение",           f"{m['rms']:.2f}"),
    ]
    w = max(len(r[0]) for r in rows)
    for label, val in rows:
        print(f"  {label:<{w}} : {val}")
    if m.get("no_contact"):
        print("  ВНИМАНИЕ: контакт не обнаружен — использованы все точки")


# ══════════════════════════════════════════════════════════════════════════════
# X11-изоляция устройства
# ══════════════════════════════════════════════════════════════════════════════

def _run_quietly(cmd: list[str]):
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _xkb_layout() -> str | None:
    try:
        out = subprocess.check_output(
            ["setxkbmap", "-query"], stderr=subprocess.DEVNULL, text=True
        )
        for line in out.splitlines():
            if line.startswith("layout:"):
                return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _set_xkb_layout(layout: str | None):
    if layout:
        _run_quietly(["setxkbmap", layout])


def _find_xinput_id(name_substr: str) -> int | None:
    try:
        out = subprocess.check_output(
            ["xinput", "list", "--short"], stderr=subprocess.DEVNULL, text=True
        )
        for line in out.splitlines():
            if name_substr.upper() in line.upper() and "Master" not in line:
                for token in line.strip().split():
                    if token.startswith("id=") and token[3:].isdigit():
                        return int(token[3:])
    except (OSError, subprocess.SubprocessError):
        pass
    return None


@contextlib.contextmanager
def xinput_isolated(name_substr: str, keep_disabled: list[bool] | None = None):
    """Контекст-менеджер: отключает xinput-устройство на время сессии.

    При выходе восстанавливает устройство и раскладку клавиатуры.
    Если keep_disabled передан и keep_disabled[0] == True —
    устройство остаётся отключённым (используется кнопкой "Завершить и отключить").
    """
    xid = _find_xinput_id(name_substr)
    layout = _xkb_layout()
    if xid is not None:
        _run_quietly(["xinput", "disable", str(xid)])
    try:
        yield
    finally:
        should_keep = keep_disabled is not None and keep_disabled[0]
        if xid is not None and not should_keep:
            _run_quietly(["xinput", "enable", str(xid)])
        _set_xkb_layout(layout)


# ══════════════════════════════════════════════════════════════════════════════
# Конфигурация
# ══════════════════════════════════════════════════════════════════════════════

_CFG_DEFAULTS: dict = {
    "xcode": 0, "ycode": 1, "rate": 50.0,
    "contact_timeout": 0.5, "calibrate": 2.0,
    "scale_x": 1.0, "scale_y": 1.0,
    "offset_x": 0.0, "offset_y": 0.0,
    "device": None, "max_x": None, "max_y": None,
}


def load_config(path: str | None) -> dict:
    cfg = dict(_CFG_DEFAULTS)
    if path:
        with open(path) as f:
            cfg.update(json.load(f))
    cfg["rate"] = max(float(cfg.get("rate") or 50.0), 0.1)
    cfg["contact_timeout"] = float(cfg["contact_timeout"])
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# UI — Dear PyGui
# ══════════════════════════════════════════════════════════════════════════════

def _make_line_theme(color: list[int]) -> int:
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots)
    return theme


def _make_scatter_theme(fill: list[int], marker_size: int = 3) -> int:
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvScatterSeries):
            dpg.add_theme_color(dpg.mvPlotCol_MarkerFill, fill, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline, fill, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, marker_size,
                                category=dpg.mvThemeCat_Plots)
    return theme


class StabiloUI:
    """Только Dear PyGui: строит окно, обновляет графики.
    Получает данные через DataBuffer, не знает про Device.
    """

    def __init__(self, buf: DataBuffer, cfg: dict, duration: float, out_path: str,
                 axis_range_x: tuple[float, float] | None = None,
                 axis_range_y: tuple[float, float] | None = None,
                 xinput_id: int | None = None):
        self._buf = buf
        self._cfg = cfg
        self._duration = duration
        self._out_path = out_path
        self._xinput_id = xinput_id      # для кнопки "Завершить и отключить"
        self.keep_disabled: list[bool] = [False]  # разделяется с xinput_isolated

        # Начальные границы — из аппаратного диапазона устройства (axis_info),
        # умноженного на scale.  Если axis_range не передан — просто запоминаем,
        # что нужно определить из первых данных.
        self._x_min = self._x_max = 0.0
        self._y_min = self._y_max = 0.0
        has_x = axis_range_x is not None
        has_y = axis_range_y is not None
        if has_x:
            raw_min, raw_max = axis_range_x
            sx = cfg["scale_x"]
            cx = (raw_min + raw_max) / 2 * sx + cfg["offset_x"]
            half = (raw_max - raw_min) / 2 * sx * 0.6 or 20.0
            self._x_min, self._x_max = cx - half, cx + half
        if has_y:
            raw_min, raw_max = axis_range_y
            sy = cfg["scale_y"]
            cy = (raw_min + raw_max) / 2 * sy + cfg["offset_y"]
            half = (raw_max - raw_min) / 2 * sy * 0.6 or 20.0
            self._y_min, self._y_max = cy - half, cy + half
        self._bounds_init = has_x or has_y

    # ── построение интерфейса ─────────────────────────────────────────────────

    def _build(self):
        th_traj_line   = _make_line_theme(COLOR_TRAJ_LINE)
        th_traj_pts    = _make_scatter_theme(COLOR_TRAJ_PTS, marker_size=3)
        th_cursor      = _make_scatter_theme(COLOR_CURSOR, marker_size=8)
        th_x_line      = _make_line_theme(COLOR_X_LINE)
        th_y_line      = _make_line_theme(COLOR_Y_LINE)

        with dpg.window(label="Stabilogram MERA", tag="main_win",
                        no_title_bar=False, no_resize=False,
                        no_move=False, no_close=True):

            dpg.add_text("Stabilogram MERA", tag="title_text", color=COLOR_TITLE)

            # ── строка статуса ─────────────────────────────────────────────
            with dpg.group(horizontal=True):
                dpg.add_text("Waiting...", tag="info_text")
                dpg.add_text("[*]", tag="contact_led",
                             color=COLOR_CONTACT_OFF)
                dpg.add_text("[Q] Stop  ")
                dpg.add_button(label="Stop", callback=self._on_stop,
                               width=80, height=28)
                dpg.add_button(label="Finish & Disable device",
                               callback=self._on_stop_keep_disabled,
                               width=200, height=28)

            # ── две колонки ───────────────────────────────────────────────
            with dpg.group(horizontal=True):
                # Левая: траектория ЦД
                with dpg.plot(label="COP trajectory", tag="traj_plot",
                              height=520, width=520, no_title=True):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="X", tag="traj_x")
                    dpg.add_plot_axis(dpg.mvYAxis, label="Y", tag="traj_y")

                    dpg.add_line_series([], [], label="trajectory",
                                       tag="traj_line", parent="traj_y")
                    dpg.bind_item_theme("traj_line", th_traj_line)

                    dpg.add_scatter_series([], [], label="points",
                                          tag="traj_pts", parent="traj_y")
                    dpg.bind_item_theme("traj_pts", th_traj_pts)

                    dpg.add_scatter_series([], [], label="current",
                                          tag="traj_cursor", parent="traj_y")
                    dpg.bind_item_theme("traj_cursor", th_cursor)

                # Правая: колонка временного ряда + метрики
                with dpg.group():
                    with dpg.plot(label="X / Y  vs  time", tag="time_plot",
                                  height=310, width=700, no_title=True):
                        dpg.add_plot_legend()
                        dpg.add_plot_axis(dpg.mvXAxis, label="time, s", tag="time_x")
                        dpg.add_plot_axis(dpg.mvYAxis, label="value",   tag="time_y")

                        dpg.add_line_series([], [], label="X",
                                            tag="x_line", parent="time_y")
                        dpg.bind_item_theme("x_line", th_x_line)

                        dpg.add_line_series([], [], label="Y",
                                            tag="y_line", parent="time_y")
                        dpg.bind_item_theme("y_line", th_y_line)

                    dpg.add_separator()

                    # Таблица метрик в реальном времени
                    with dpg.table(tag="metrics_table", header_row=True,
                                   borders_innerH=True, borders_outerH=True,
                                   borders_innerV=True, borders_outerV=True,
                                   width=700):
                        dpg.add_table_column(label="Metric")
                        dpg.add_table_column(label="Value")

                        _metric_rows = [
                            ("sigma X / sigma Y",    "metrics_sxsy"),
                            ("Range X / Y",          "metrics_range"),
                            ("Path length L",        "metrics_L"),
                            ("Mean velocity",        "metrics_vel"),
                            ("RMS",                  "metrics_rms"),
                            ("Mean X / Y",           "metrics_mean"),
                        ]
                        for label, tag in _metric_rows:
                            with dpg.table_row():
                                dpg.add_text(label)
                                dpg.add_text("-", tag=tag)

    # ── обратные вызовы ────────────────────────────────────────────────────────

    def _on_stop(self, *args):
        """Останавливает запись и восстанавливает xinput."""
        self._buf.stop()
        dpg.stop_dearpygui()

    def _on_stop_keep_disabled(self, *args):
        """Останавливает запись, xinput остаётся отключённым."""
        self.keep_disabled[0] = True
        self._buf.stop()
        dpg.stop_dearpygui()

    def _on_key_press(self, sender, app_data):
        """Горячие клавиши: Q / Escape — стоп."""
        if app_data in (dpg.mvKey_Q, dpg.mvKey_Escape):
            self._on_stop()

    # ── кадровое обновление ────────────────────────────────────────────────────

    def _update_trajectory(self, arr: np.ndarray):
        t, x, y, c = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        sel = c > 0.5
        xs, ys = (x[sel], y[sel]) if sel.any() else (x, y)

        if len(xs) == 0:
            return

        dpg.set_value("traj_line",   [xs.tolist(), ys.tolist()])
        dpg.set_value("traj_pts",    [xs.tolist(), ys.tolist()])
        dpg.set_value("traj_cursor", [[float(xs[-1])], [float(ys[-1])]])

        # Масштаб: берём максимальный охват за всю сессию.
        # Минимальный полуразмах окна — 20 единиц (чтобы при малых колебаниях
        # не было чрезмерного увеличения).
        MIN_HALF = 20.0
        span_x = max((xs.max() - xs.min()) * 0.55, MIN_HALF)
        span_y = max((ys.max() - ys.min()) * 0.55, MIN_HALF)
        # Центр — текущее среднее
        cx, cy = float(xs.mean()), float(ys.mean())
        new_xmin, new_xmax = cx - span_x, cx + span_x
        new_ymin, new_ymax = cy - span_y, cy + span_y

        if not self._bounds_init:
            self._x_min, self._x_max = new_xmin, new_xmax
            self._y_min, self._y_max = new_ymin, new_ymax
            self._bounds_init = True
        else:
            # Только расширяем, не сужаем — чтобы не прыгал масштаб
            self._x_min = min(self._x_min, new_xmin)
            self._x_max = max(self._x_max, new_xmax)
            self._y_min = min(self._y_min, new_ymin)
            self._y_max = max(self._y_max, new_ymax)

        dpg.set_axis_limits("traj_x", self._x_min, self._x_max)
        dpg.set_axis_limits("traj_y", self._y_min, self._y_max)

        # Индикатор контакта
        in_contact = bool(c[-1] > 0.5)
        dpg.configure_item("contact_led",
                           color=COLOR_CONTACT_ON if in_contact else COLOR_CONTACT_OFF)

    def _update_timeseries(self, arr: np.ndarray):
        t, x, y = arr[:, 0], arr[:, 1], arr[:, 2]
        if len(t) == 0:
            return
        t_max = t[-1]
        mask = t > (t_max - WINDOW_SEC)
        if not mask.any():
            return
        tm, xm, ym = t[mask], x[mask], y[mask]
        dpg.set_value("x_line", [tm.tolist(), xm.tolist()])
        dpg.set_value("y_line", [tm.tolist(), ym.tolist()])
        dpg.set_axis_limits("time_x", float(tm[0]), float(tm[-1]) + 0.1)
        vals = np.concatenate([xm, ym])
        if len(vals):
            val_range = vals.max() - vals.min()
            # Минимальный полурасстояние — 5% от полного диапазона оси или 2 единицы
            v_margin = max(val_range * 0.15, 2.0)
            dpg.set_axis_limits("time_y",
                                float(vals.min()) - v_margin,
                                float(vals.max()) + v_margin)

    def _update_metrics(self, arr: np.ndarray):
        m = compute_metrics(arr)
        if not m:
            return
        dpg.set_value("metrics_sxsy",  f"{m['sx']:.2f} / {m['sy']:.2f}")
        dpg.set_value("metrics_range", f"{m['range_x']:.2f} / {m['range_y']:.2f}")
        dpg.set_value("metrics_L",     f"{m['L']:.2f}")
        dpg.set_value("metrics_vel",   f"{m['vel']:.2f} u/s")
        dpg.set_value("metrics_rms",   f"{m['rms']:.2f}")
        dpg.set_value("metrics_mean",  f"{m['xm']:.2f} / {m['ym']:.2f}")

    def _update_status(self, arr: np.ndarray):
        if len(arr) == 0:
            dpg.set_value("info_text", "Waiting for data...")
            return
        t, c = arr[:, 0], arr[:, 3]
        n_total   = len(arr)
        n_contact = int((c > 0.5).sum())
        pct = 100 * n_contact / n_total if n_total else 0
        rate = self._buf._cfg["rate"]
        dpg.set_value("info_text",
                      f"Samples: {n_total} | Contact: {n_contact} ({pct:.0f}%) | "
                      f"T={t[-1]:.1f}s | {rate:.0f} Hz  ")

    def update_frame(self):
        arr = self._buf.snapshot()

        if arr.size > 0:
            self._update_trajectory(arr)
            self._update_timeseries(arr)
            self._update_metrics(arr)
        self._update_status(arr)

        # Авто-стоп по длительности
        if self._duration and self._buf.elapsed > self._duration:
            self._on_stop()

    # ── главный цикл ──────────────────────────────────────────────────────────

    @staticmethod
    def _load_font():
        """Загружает первый найденный TTF-шрифт с кириллическим диапазоном.
        DearPyGui 2.3+ автоматически включает все диапазоны символов шрифта.
        """
        font_path = next((p for p in _FONT_CANDIDATES if os.path.exists(p)), None)
        if font_path is None:
            return  # останется встроенный шрифт (только ASCII)
        with dpg.font_registry():
            font = dpg.add_font(font_path, 16)
        dpg.bind_font(font)

    def run(self) -> tuple[np.ndarray, dict | None]:
        dpg.create_context()
        dpg.create_viewport(title="Stabilogram MERA", width=1280, height=660)
        dpg.setup_dearpygui()

        self._load_font()
        self._build()

        # Горячие клавиши — на уровне stage (DearPyGui 2.x требует mvStage)
        with dpg.handler_registry():
            dpg.add_key_press_handler(dpg.mvKey_Q,      callback=self._on_key_press)
            dpg.add_key_press_handler(dpg.mvKey_Escape,  callback=self._on_key_press)

        dpg.set_primary_window("main_win", True)
        dpg.show_viewport()

        # Поток дискретизации
        sample_thread = threading.Thread(
            target=self._buf.sample_loop, daemon=True
        )
        sample_thread.start()

        while dpg.is_dearpygui_running() and self._buf.running:
            self.update_frame()
            dpg.render_dearpygui_frame()
            time.sleep(0.02)

        self._buf.stop()
        dpg.destroy_context()
        return self._finalize()

    def _finalize(self) -> tuple[np.ndarray, dict | None]:
        arr = self._buf.to_array()
        m   = compute_metrics(arr)
        save_csv(arr, self._out_path)
        print_summary(m)
        print(f"\nСохранён CSV: {self._out_path}")
        if m:
            print(f"Длительность: {m['T']:.2f} с, отсчётов: {m['n']}")
        return arr, m


# ══════════════════════════════════════════════════════════════════════════════
# Headless-режим
# ══════════════════════════════════════════════════════════════════════════════

def run_headless(buf: DataBuffer, duration: float, out_path: str):
    print("Headless-режим: Ctrl-C для остановки...")
    sample_thread = threading.Thread(target=buf.sample_loop, daemon=True)
    sample_thread.start()
    try:
        while buf.running:
            if duration and buf.elapsed > duration:
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    buf.stop()
    arr = buf.to_array()
    m = compute_metrics(arr)
    save_csv(arr, out_path)
    print_summary(m)
    print(f"\nСохранён CSV: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Точка входа
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Стабилограмма MERA (Dear PyGui)")
    ap.add_argument("--device",   help="путь к /dev/input/eventN")
    ap.add_argument("--axes",     help="JSON-конфиг осей и калибровки")
    ap.add_argument("--rate",     type=float, default=0,  help="частота, Гц")
    ap.add_argument("--duration", type=float, default=0,  help="авто-стоп, с (0=бесконечно)")
    ap.add_argument("--out",      default="stabilo.csv",  help="выходной CSV")
    ap.add_argument("--no-gui",   action="store_true",    help="headless-режим")
    ap.add_argument("command",    nargs="?", default="live",
                    choices=["live", "list"], help="режим")
    args = ap.parse_args()

    cfg = load_config(args.axes)
    if args.device:
        cfg["device"] = args.device
    if args.rate > 0:
        cfg["rate"] = args.rate

    if args.command == "list":
        d = discover_device()
        print("Найдено устройство:", d or "не найдено")
        return

    path = cfg["device"] or discover_device()
    if not path:
        print("Устройство MERA не найдено. Укажите --device /dev/input/eventN",
              file=sys.stderr)
        sys.exit(1)

    if not is_likely_stabilo_platform(path):
        print(f"Предупреждение: {path} похож на тачпад или мышь.", file=sys.stderr)

    try:
        dev = Device(path, max_x=cfg.get("max_x"), max_y=cfg.get("max_y"))
        dev.start()
    except OSError as e:
        print(f"Не удалось открыть {path}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Устройство: {path}")
    print(f"Калибровка осей ({cfg['calibrate']} с)...")
    xcode, ycode = auto_detect_axes(dev, cfg["calibrate"])
    print(f"Оси: X=code {xcode}, Y=code {ycode}")

    buf = DataBuffer(cfg, dev, xcode, ycode)

    if args.no_gui:
        run_headless(buf, args.duration, args.out)
        dev.stop_now()
        dev.close()
        return

    # Передаём аппаратный диапазон осей для корректного начального масштаба
    ax_range_x = dev.axis_info.get(xcode)  # (min, max) или None
    ax_range_y = dev.axis_info.get(ycode)
    xinput_id  = _find_xinput_id("MERA")

    ui = StabiloUI(
        buf,
        cfg=cfg,
        duration=args.duration,
        out_path=args.out,
        axis_range_x=ax_range_x,
        axis_range_y=ax_range_y,
        xinput_id=xinput_id,
    )
    with xinput_isolated("MERA", keep_disabled=ui.keep_disabled):
        ui.run()

    dev.stop_now()
    dev.close()


if __name__ == "__main__":
    main()
