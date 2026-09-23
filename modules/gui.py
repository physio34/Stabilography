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
import json
import os
import re
import select
import struct
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd
import dearpygui.dearpygui as dpg

try:
    from modules.config import resolve_out_path
except ImportError:  # python -m modules.gui / прямой запуск
    from .config import resolve_out_path

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


if sys.platform.startswith('win'):
    # Логика для Windows (без fcntl)
    def lock_file(fd):
        print("Блокировка файла не поддерживается на Windows этим способом")
else:
    import fcntl
    def lock_file(fd):
        fcntl.flock(fd, fcntl.LOCK_EX)


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
    """Ищет устройство MERA: Linux — /sys/class/input, Windows — COM/HID."""
    if sys.platform == "win32":
        return _discover_device_windows()
    return _discover_device_linux()


def _discover_device_linux() -> str | None:
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


def _discover_device_windows() -> str | None:
    try:
        import serial.tools.list_ports
        ports = serial.tools.list_ports.comports()
        for port in ports:
            desc = (port.description or "").upper()
            mfg = (port.manufacturer or "").upper()
            hwid = (port.hwid or "").upper()
            if "MERA" in desc or "MERA" in mfg:
                return port.device
            if "VID_" in hwid and "PID_" in hwid:
                if _is_mera_vid_pid(hwid):
                    return port.device
    except ImportError:
        pass

    try:
        import hid
        for d in hid.enumerate():
            prod = (d.get("product_string") or "").upper()
            mfg = (d.get("manufacturer_string") or "").upper()
            if "MERA" in prod or "MERA" in mfg:
                return f"hid://{d['vendor_id']:04x}:{d['product_id']:04x}"
    except ImportError:
        pass

    return None


def _is_mera_vid_pid(hwid: str) -> bool:
    known = {
        "VID_0483", "VID_2342", "VID_1FC9", "VID_03EB",
        "VID_16C0", "VID_0403", "VID_2A03",
    }
    for part in hwid.replace(";", " ").split():
        up = part.upper()
        if up.startswith("VID_") and up in known:
            return True
    return False


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
    """Низкоуровневое чтение evdev/HID/serial + валидация значений."""

    def __init__(self, path: str, max_x: int | None = None, max_y: int | None = None):
        self._path = path
        self.max_x = max_x
        self.max_y = max_y
        self._raw: collections.deque = collections.deque(maxlen=400_000)
        self._latest: dict = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.valid_codes: set[int] = set()
        self.axis_info: dict[int, tuple[int, int]] = {}
        self._backend: str | None = None
        self._serial = None
        self._hid = None
        self._prev_x: int | None = None
        self._prev_y: int | None = None
        self._accum_x: int = 0
        self._accum_y: int = 0

        if sys.platform == "win32" and not path.startswith("/dev/"):
            self._init_windows(path)
        else:
            self._init_linux(path)

    def _init_linux(self, path: str):
        self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        self._poll = select.poll()
        self._poll.register(self.fd, select.POLLIN)
        self._backend = "linux"
        self.valid_codes = set(_get_abs_capabilities(path) or [])
        for code in list(self.valid_codes):
            try:
                _, mn, mx, _, _, _ = _eviocgabs(self.fd, code)
                self.axis_info[code] = (mn, mx)
            except (IOError, OSError, ValueError):
                pass

    def _init_windows(self, path: str):
        if path.startswith("hid://"):
            import re
            m = re.search(r"hid://([0-9a-fA-F]+):([0-9a-fA-F]+)", path)
            if not m:
                print(f"Неверный формат HID-адреса: {path} (ожидается hid://VID:PID)", file=sys.stderr)
                sys.exit(1)
            vid, pid = int(m.group(1), 16), int(m.group(2), 16)
            import hid
            self._hid = hid.device()
            self._hid.open(vid, pid)
            self._hid.set_nonblocking(True)
            self._backend = "hid"
            return

        try:
            import serial
            self._serial = serial.Serial(path, baudrate=115200, timeout=0)
            self._backend = "serial"
        except ImportError:
            print("Для работы на Windows установите pyserial: pip install pyserial", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Не удалось открыть {path}: {e}", file=sys.stderr)
            sys.exit(1)

    def start(self):
        self._thread.start()

    def stop_now(self):
        self._stop.set()

    def close(self):
        if self._backend == "hid" and self._hid:
            try:
                self._hid.close()
            except Exception:
                pass
        elif self._backend == "serial" and self._serial:
            try:
                self._serial.close()
            except Exception:
                pass
        elif self._backend == "linux":
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
        if self._backend == "hid":
            self._run_hid()
        elif self._backend == "serial":
            self._run_serial()
        else:
            self._run_linux()

    def _run_linux(self):
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

    def _run_hid(self):
        buf_size = 64
        while not self._stop.is_set():
            try:
                data = self._hid.read(buf_size)
            except Exception:
                time.sleep(0.01)
                continue
            if not data:
                time.sleep(0.01)
                continue
            self._parse_hdev_packet(bytes(data))

    def _run_serial(self):
        buf = bytearray()
        while not self._stop.is_set():
            try:
                n = self._serial.in_waiting
                if n > 0:
                    buf.extend(self._serial.read(n))
                else:
                    time.sleep(0.005)
                    continue
            except Exception:
                time.sleep(0.01)
                continue
            while len(buf) >= 8:
                if buf[0] == 0xAA and len(buf) >= 8:
                    packet = bytes(buf[:8])
                    del buf[:8]
                    self._parse_hdev_packet(packet)
                else:
                    buf.pop(0)

    def _parse_hdev_packet(self, data: bytes):
        now = time.time()
        if len(data) < 64:
            return
        raw_x, raw_y = data[1], data[2]
        contact = 1 if data[63] == 0x40 else 0

        if self._prev_x is not None:
            dx = raw_x - self._prev_x
            if dx > 128:
                dx -= 256
            elif dx < -128:
                dx += 256
            self._accum_x += dx
        else:
            self._accum_x = raw_x
        self._prev_x = raw_x

        if self._prev_y is not None:
            dy = raw_y - self._prev_y
            if dy > 128:
                dy -= 256
            elif dy < -128:
                dy += 256
            self._accum_y += dy
        else:
            self._accum_y = raw_y
        self._prev_y = raw_y

        with self._lock:
            self._latest[0] = (self._accum_x, now)
            self._latest[1] = (self._accum_y, now)
            self._raw.append((now, 0, self._accum_x))
            self._raw.append((now, 1, self._accum_y))


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


def save_csv(arr: np.ndarray, path: str, meta: dict | None = None,
             t_start: float | None = None, t_end: float | None = None):
    if not path:
        return
    import datetime
    from pathlib import Path as _Path
    _Path(path).parent.mkdir(parents=True, exist_ok=True)
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
        for row in arr:
            f.write(f"{row[0]:.3f},{row[1]:.3f},{row[2]:.3f}\n")


def _build_csv_path(base_path: str, meta: dict | None = None,
                    t_start: float | None = None) -> str:
    """Создаёт имя CSV-файла: base_YYYYMMDD_HHMMSS[_ID][_G][_Age].csv"""
    import datetime
    import os
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
    "axis_zero": 128,
    "unit_scale": 0.1,
    "median_window": 3,
    "outlier_sigma": 5.0,
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
                 xinput_id: int | None = None,
                 meta: dict | None = None):
        self._buf = buf
        self._cfg = cfg
        self._duration = duration
        self._out_path = out_path
        self._xinput_id = xinput_id
        self.keep_disabled: list[bool] = [False]
        self.meta = meta or {}

        # Начальные границы — из аппаратного диапазона устройства (axis_info),
        # умноженного на scale.  Если axis_range не передан — просто запоминаем,
        # что нужно определить из первых данных.
        self._x_min = self._x_max = 0.0
        self._y_min = self._y_max = 0.0
        has_x = axis_range_x is not None
        has_y = axis_range_y is not None
        az = cfg["axis_zero"]
        us = cfg["unit_scale"]
        if has_x:
            raw_min, raw_max = axis_range_x
            sx = cfg["scale_x"]
            cx = ((raw_min + raw_max) / 2 - az) * sx * us + cfg["offset_x"]
            half = (raw_max - raw_min) / 2 * sx * us * 0.6 or 0.2
            self._x_min, self._x_max = cx - half, cx + half
        if has_y:
            raw_min, raw_max = axis_range_y
            sy = cfg["scale_y"]
            cy = ((raw_min + raw_max) / 2 - az) * sy * us + cfg["offset_y"]
            half = (raw_max - raw_min) / 2 * sy * us * 0.6 or 0.2
            self._y_min, self._y_max = cy - half, cy + half
        self._bounds_init = has_x or has_y

    # ── построение интерфейса ─────────────────────────────────────────────────

    def _build(self):
        th_traj_line   = _make_line_theme(COLOR_TRAJ_LINE)
        th_traj_pts    = _make_line_theme(COLOR_TRAJ_PTS)
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

                    dpg.add_line_series([], [], label="points",
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
        MIN_HALF = 0.2
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
            v_margin = max(val_range * 0.15, 0.02)
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

    def _build_meta_dialog(self):
        """Модальное окно ввода метаданных (ID, пол, возраст)."""
        with dpg.window(label="Patient Data", tag="meta_win",
                        modal=True, no_close=True,
                        width=350, height=220, pos=[460, 220]):
            dpg.add_text("Enter patient data:")
            dpg.add_input_text(label="ID",   tag="meta_id",   width=250)
            dpg.add_combo(["M", "F"], label="Gender", tag="meta_gender",
                          default_value="M", width=250)
            dpg.add_input_int(label="Age", tag="meta_age",
                              default_value=30, min_value=0, max_value=150,
                              width=250)
            dpg.add_button(label="Start", callback=self._on_meta_submit,
                           width=250, height=30)

    def _on_meta_submit(self, *args):
        self.meta = {
            "id":     dpg.get_value("meta_id"),
            "gender": dpg.get_value("meta_gender"),
            "age":    dpg.get_value("meta_age"),
        }
        dpg.delete_item("meta_win")

    def run(self) -> tuple[np.ndarray, dict | None]:
        dpg.create_context()
        dpg.create_viewport(title="Stabilogram MERA", width=1280, height=660)
        dpg.setup_dearpygui()

        self._load_font()
        self._build_meta_dialog()
        self._build()

        # Горячие клавиши — на уровне stage
        with dpg.handler_registry():
            dpg.add_key_press_handler(dpg.mvKey_Q,      callback=self._on_key_press)
            dpg.add_key_press_handler(dpg.mvKey_Escape,  callback=self._on_key_press)

        dpg.set_primary_window("main_win", False)
        dpg.show_viewport()

        # Показываем модальный диалог, ждём нажатия Start
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
            if not dpg.does_item_exist("meta_win"):
                break
            time.sleep(0.02)

        # После закрытия диалога — запускаем основное окно
        dpg.set_primary_window("main_win", True)

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
        t_start = self._buf._start
        t_end = time.time()
        out_path = _build_csv_path(self._out_path, meta=self.meta, t_start=t_start)
        save_csv(arr, out_path, meta=self.meta, t_start=t_start, t_end=t_end)
        print_summary(m)
        print(f"\nCSV: {out_path}")
        if self.meta:
            print(f"Patient: {self.meta.get('id', '-')}, "
                  f"{self.meta.get('gender', '-')}, "
                  f"{self.meta.get('age', '-')} y.o.")
        if m:
            print(f"Duration: {m['T']:.2f} s, samples: {m['n']}")
        return arr, m


# ══════════════════════════════════════════════════════════════════════════════
# Headless-режим
# ══════════════════════════════════════════════════════════════════════════════

def run_headless(buf: DataBuffer, duration: float, out_path: str,
                 meta: dict | None = None):
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
    t_start = buf._start
    t_end = time.time()
    out_path = _build_csv_path(out_path, meta=meta, t_start=t_start)
    save_csv(arr, out_path, meta=meta, t_start=t_start, t_end=t_end)
    print_summary(m)
    print(f"\nCSV: {out_path}")


def _cmd_sniff(args):
    """Открывает HID-устройство и выводит сырые пакеты в hex."""
    import re
    path = args.device
    if not path:
        path = discover_device()
    if not path:
        print("Устройство не найдено.", file=sys.stderr)
        sys.exit(1)
    print(f"Устройство: {path}")

    if path.startswith("hid://"):
        import hid as hid_mod
        m = re.search(r"hid://([0-9a-fA-F]+):([0-9a-fA-F]+)", path)
        if not m:
            print(f"Неверный HID-адрес: {path}", file=sys.stderr)
            sys.exit(1)
        vid, pid = int(m.group(1), 16), int(m.group(2), 16)
        dev = hid_mod.device()
        dev.open(vid, pid)
        dev.set_nonblocking(True)
        print(f"HID открыт: VID={vid:04x} PID={pid:04x}")
        print("Сырые пакеты (Ctrl-C для остановки):\n")
        try:
            pkt_count = 0
            while True:
                data = dev.read(64)
                if data:
                    pkt_count += 1
                    raw = bytes(data)
                    hex_str = raw.hex(" ")
                    ascii_str = "".join(chr(b) if 32 <= b < 127 else "." for b in raw)
                    print(f"[{pkt_count:5d}] len={len(raw):2d}  hex={hex_str}")
                    print(f"       ascii={ascii_str}")
                    if pkt_count <= 10:
                        print(f"       bin={raw!r}")
                else:
                    import time
                    time.sleep(0.01)
        except KeyboardInterrupt:
            print(f"\nВсего пакетов: {pkt_count}")
        finally:
            dev.close()
    else:
        print(f"sniff пока поддерживает только HID-устройства (получен: {path})")


# ══════════════════════════════════════════════════════════════════════════════
# Точка входа
# ══════════════════════════════════════════════════════════════════════════════

def main(argv=None):
    ap = argparse.ArgumentParser(description="Стабилограмма MERA (Dear PyGui)")
    ap.add_argument("--device",   help="путь к устройству (Linux: /dev/input/eventN, Windows: COM-порт или hid://VID:PID)")
    ap.add_argument("--axes",     help="JSON-конфиг осей и калибровки")
    ap.add_argument("--rate",     type=float, default=0,  help="частота, Гц")
    ap.add_argument("--duration", type=float, default=0,  help="авто-стоп, с (0=бесконечно)")
    ap.add_argument("--out",      default="stabilo.csv",  help="выходной CSV (имя без пути -> каталог records/)")
    ap.add_argument("--no-gui",   action="store_true",    help="headless-режим")
    ap.add_argument("--id",       default="", help="ID пациента")
    ap.add_argument("--gender",   default="", help="пол (M/F)")
    ap.add_argument("--age",      type=int, default=0,    help="возраст")
    ap.add_argument("--unit-scale", type=float, default=0, help="масштаб единиц (0.01=деление на 100, мм)")
    ap.add_argument("--median",   type=int, default=0,    help="окно медианного фильтра (0=3)")
    ap.add_argument("--outlier",  type=float, default=0,  help="порог выбросов в sigma (0=5.0)")
    ap.add_argument("--debug",    action="store_true",    help="hex-dump сырых HID-пакетов")
    ap.add_argument("command",    nargs="?", default="live",
                    choices=["live", "list", "sniff"], help="режим: live/list/sniff")
    args = ap.parse_args(argv)
    args.out = str(resolve_out_path(args.out))

    cfg = load_config(args.axes)
    if args.device:
        cfg["device"] = args.device
    if args.rate > 0:
        cfg["rate"] = args.rate
    if args.unit_scale > 0:
        cfg["unit_scale"] = args.unit_scale
    if args.median > 0:
        cfg["median_window"] = args.median
    if args.outlier > 0:
        cfg["outlier_sigma"] = args.outlier

    if args.command == "sniff":
        _cmd_sniff(args)
        return

    if args.command == "list":
        d = discover_device()
        print("Найдено устройство:", d or "не найдено")
        return

    path = cfg["device"] or discover_device()
    if not path:
        if sys.platform == "win32":
            print("Устройство MERA не найдено. Укажите --device COM3 (или другой COM-порт)", file=sys.stderr)
        else:
            print("Устройство MERA не найдено. Укажите --device /dev/input/eventN", file=sys.stderr)
        sys.exit(1)

    if sys.platform != "win32" and not is_likely_stabilo_platform(path):
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

    meta = {"id": args.id, "gender": args.gender, "age": args.age}

    if args.no_gui:
        run_headless(buf, args.duration, args.out, meta=meta)
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
        meta=meta,
    )
    with xinput_isolated("MERA", keep_disabled=ui.keep_disabled):
        ui.run()

    dev.stop_now()
    dev.close()


if __name__ == "__main__":
    main()
