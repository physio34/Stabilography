import argparse
import collections
import json
import os
import platform
import re
import select
import struct
import sys
import threading
import time

import numpy as np
import matplotlib
import pandas as pd

if os.environ.get("DISPLAY") or sys.platform == "win32":
    matplotlib.use("TkAgg")
else:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

try:
    from modules.config import resolve_out_path
except ImportError:  # python -m modules.record / прямой запуск
    from .config import resolve_out_path

EV_ABS = 3
EV_SYN = 0
EVENT_FMT = "llHHi"
EVENT_SIZE = struct.calcsize(EVENT_FMT)


def discover_device():
    if sys.platform == "win32":
        return _discover_device_windows()
    return _discover_device_linux()


def _discover_device_linux():
    base = "/sys/class/input"
    try:
        names = os.listdir(base)
    except OSError:
        return None
    for name in names:
        if not name.startswith("event"):
            continue
        p = os.path.join(base, name, "device", "name")
        try:
            with open(p) as f:
                devname = f.read().strip()
        except OSError:
            continue
        if "MERA" in devname.upper():
            return os.path.join("/dev/input", name)
    return None


def _discover_device_windows():
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


def _is_mera_vid_pid(hwid):
    known = {
        "VID_0483", "VID_2342", "VID_1FC9", "VID_03EB",
        "VID_16C0", "VID_0403", "VID_2A03",
    }
    for part in hwid.replace(";", " ").split():
        up = part.upper()
        if up.startswith("VID_") and up in known:
            return True
    return False


def get_device_capabilities(dev_path):
    """Read device capabilities to verify it's a stabilo platform, not a touchpad."""
    base = dev_path.replace("/dev/input/", "/sys/class/input/")
    cap_path = os.path.join(base, "device", "capabilities", "abs")
    try:
        with open(cap_path) as f:
            cap_hex = f.read().strip()
        cap_val = int(cap_hex, 16)
        codes = []
        for i in range(64):
            if cap_val & (1 << i):
                codes.append(i)
        return codes
    except (OSError, ValueError):
        return None


def is_likely_stabilo_platform(dev_path):
    """
    Check if device is likely a stabilo platform and not a touchpad.
    Touchpads typically support many ABS codes (multitouch: 47-63+),
    while stabilo platforms usually only have 2-3 codes (X, Y, pressure).
    """
    caps = get_device_capabilities(dev_path)
    if caps is None:
        return True  # Can't verify, assume ok if name matched

    # Touchpads have many capabilities (multitouch, etc.)
    if len(caps) > 5:
        return False

    # Check for multitouch codes (ABS_MT_SLOT=47, ABS_MT_TOUCH_MAJOR=48, etc.)
    mt_codes = set(range(47, 64))
    if any(c in mt_codes for c in caps):
        return False

    return True


def load_axes_config(path):
    cfg = {
        "xcode": 0,
        "ycode": 1,
        "rate": 50,
        "contact_timeout": 0.5,
        "calibrate": 2.0,
        "scale_x": 0.1,
        "scale_y": 0.1,
        "offset_x": 0.0,
        "offset_y": 0.0,
        "device": None,
        "max_x": None,
        "max_y": None,
    }
    if path:
        with open(path) as f:
            cfg.update(json.load(f))
    return cfg


class Device:
    def __init__(self, path, max_x=None, max_y=None):
        self.path = path
        self.max_x = max_x
        self.max_y = max_y
        self.valid_codes = set()
        self.raw = collections.deque(maxlen=400000)
        self.latest = {}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self._backend = None
        self._serial = None
        self._hid = None
        self._prev_x = None
        self._prev_y = None
        self._accum_x = 0
        self._accum_y = 0

        if sys.platform == "win32" and not path.startswith("/dev/"):
            self._init_windows(path)
        else:
            self._init_linux(path)

    def _init_linux(self, path):
        self.fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        self.poll = select.poll()
        self.poll.register(self.fd, select.POLLIN)
        caps = get_device_capabilities(path)
        if caps:
            self.valid_codes = set(caps)

    def _init_windows(self, path):
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
        self.thread.start()

    def stop_now(self):
        self.stop.set()

    def _validate_value(self, code, val):
        """Validate that the value is within expected range for stabilo platform."""
        if self.valid_codes and code not in self.valid_codes:
            return False
        if self.max_x is not None and code == 0 and val > self.max_x:
            return False
        if self.max_y is not None and code == 1 and val > self.max_y:
            return False
        # Touchpads often have very high resolution values
        if val > 65535:
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
        while not self.stop.is_set():
            evs = self.poll.poll(100)
            if not evs:
                continue
            try:
                data = os.read(self.fd, EVENT_SIZE * 64)
            except OSError:
                continue
            now = time.time()
            for off in range(0, len(data) - EVENT_SIZE + 1, EVENT_SIZE):
                sec, usec, typ, code, val = struct.unpack(
                    EVENT_FMT, data[off : off + EVENT_SIZE]
                )
                if typ == EV_ABS:
                    if not self._validate_value(code, val):
                        continue
                    with self.lock:
                        self.latest[code] = (val, now)
                        self.raw.append((now, code, val))

    def _run_hid(self):
        buf_size = 64
        while not self.stop.is_set():
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
        while not self.stop.is_set():
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

    def _parse_hdev_packet(self, data):
        now = time.time()
        if len(data) < 64:
            return
        raw_x, raw_y = data[1], data[2]

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

        with self.lock:
            self.latest[0] = (self._accum_x, now)
            self.latest[1] = (self._accum_y, now)
            self.raw.append((now, 0, self._accum_x))
            self.raw.append((now, 1, self._accum_y))

    def recent_codes(self, since):
        codes = {}
        with self.lock:
            items = list(self.raw)
        for t, code, val in items:
            if t >= since:
                codes.setdefault(code, []).append(val)
        return codes

    def get(self, code):
        with self.lock:
            return self.latest.get(code, (None, 0.0))

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
        elif hasattr(self, "fd"):
            try:
                os.close(self.fd)
            except OSError:
                pass


def auto_detect_axes(dev, window):
    since = time.time()
    time.sleep(window)
    codes = dev.recent_codes(since)
    if not codes:
        return 0, 1
    if 0 in codes and 1 in codes:
        return 0, 1
    var = sorted(
        codes.items(),
        key=lambda kv: float(np.var(kv[1])) if len(kv[1]) > 1 else 0.0,
        reverse=True,
    )
    if len(var) >= 2:
        return var[0][0], var[1][0]
    return var[0][0], var[0][0]


def compute_metrics(arr):
    if arr.size == 0:
        return None
    t, x, y, c = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    sel = c > 0.5
    if not sel.any():
        sel = np.ones_like(c, dtype=bool)
    xx, yy, tt = x[sel], y[sel], t[sel]
    if len(xx) < 2:
        return dict(n=int(len(xx)), T=0.0, xm=float(xx.mean()) if len(xx) else 0.0,
                    ym=float(yy.mean()) if len(yy) else 0.0, sx=0.0, sy=0.0,
                    maxdev_x=0.0, maxdev_y=0.0, range_x=0.0, range_y=0.0,
                    L=0.0, vel=0.0, rms=0.0)
    xm, ym = xx.mean(), yy.mean()
    sx, sy = xx.std(), yy.std()
    maxdev_x = float(np.abs(xx - xm).max())
    maxdev_y = float(np.abs(yy - ym).max())
    range_x = float(xx.max() - xx.min())
    range_y = float(yy.max() - yy.min())
    dx, dy = np.diff(xx), np.diff(yy)
    L = float(np.sqrt(dx ** 2 + dy ** 2).sum())
    T = float(tt[-1] - tt[0])
    vel = L / T if T > 0 else 0.0
    rms = float(np.sqrt(((xx - xm) ** 2 + (yy - ym) ** 2).mean()))
    return dict(n=int(len(xx)), T=T, xm=float(xm), ym=float(ym), sx=float(sx),
                sy=float(sy), maxdev_x=maxdev_x, maxdev_y=maxdev_y,
                range_x=range_x, range_y=range_y, L=L, vel=vel, rms=rms)


def save_csv(arr, path, t_start=None, t_end=None):
    if not path:
        return
    import datetime
    from pathlib import Path as _Path
    _Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        if t_start is not None:
            utc_start = datetime.datetime.fromtimestamp(t_start, tz=datetime.timezone.utc)
            f.write(f"#Start_UTC,{utc_start.strftime('%Y-%m-%dT%H:%M:%S.%f')}Z\n")
        if t_end is not None:
            utc_end = datetime.datetime.fromtimestamp(t_end, tz=datetime.timezone.utc)
            f.write(f"#End_UTC,{utc_end.strftime('%Y-%m-%dT%H:%M:%S.%f')}Z\n")
        f.write("#t_rel_s,x,y\n")
        for row in arr:
            f.write(f"{row[0]:.5f},{row[1]:.5f},{row[2]:.5f}\n")


def _build_csv_path(base_path, t_start=None):
    """Создаёт имя CSV-файла: base_YYYYMMDD_HHMMSS.csv"""
    import datetime
    base, ext = os.path.splitext(base_path)
    parts = [base]
    if t_start is not None:
        utc = datetime.datetime.fromtimestamp(t_start, tz=datetime.timezone.utc)
        parts.append(utc.strftime("%Y%m%d_%H%M%S"))
    return "_".join(parts) + ext


def make_figure(arr, metrics):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    if arr.size:
        t, x, y, c = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        sel = c > 0.5
        if sel.any():
            ax1.plot(x[sel], y[sel], "-", lw=0.8, color="gray", alpha=0.5)
            sc = ax1.scatter(x[sel], y[sel], c=t[sel], cmap="viridis", s=8)
            ax1.plot(x[sel][0], y[sel][0], "go", label="старт")
            ax1.plot(x[sel][-1], y[sel][-1], "ro", label="финиш")
            fig.colorbar(sc, ax=ax1, label="время, с")
        ax1.set_aspect("equal")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_title("Траектория ЦД (стабилограмма)")
        ax1.legend(loc="best")
        ax2.plot(t, x, "-", label="X")
        ax2.plot(t, y, "-", label="Y")
        if len(x):
            ax2.plot(t, c * (np.nanmax(np.abs(x)) + 1), "k", alpha=0.3, label="контакт")
        ax2.set_xlabel("время, с")
        ax2.set_ylabel("значение")
        ax2.set_title("Временные ряды")
        ax2.legend(loc="best")
    else:
        ax1.set_title("Нет данных")
        ax2.set_title("Нет данных")
    if metrics:
        txt = (f"L={metrics['L']:.1f}  σx={metrics['sx']:.1f}  σy={metrics['sy']:.1f}  "
               f"maxDevX={metrics['maxdev_x']:.1f}  maxDevY={metrics['maxdev_y']:.1f}  "
               f"T={metrics['T']:.1f} с  V={metrics['vel']:.1f}")
        fig.suptitle(txt, fontsize=10)
    fig.tight_layout()
    return fig


def print_summary(m):
    if not m:
        print("Нет данных для анализа.")
        return
    print("=== Стабилографические показатели ===")
    print(f"Число отсчётов (контакт): {m['n']}")
    print(f"Длительность записи, с   : {m['T']:.2f}")
    print(f"Среднее положение X/Y     : {m['xm']:.2f} / {m['ym']:.2f}")
    print(f"Длина тракта L, ед.       : {m['L']:.2f}")
    print(f"σ по X / по Y, ед.        : {m['sx']:.2f} / {m['sy']:.2f}")
    print(f"max откл. по X / по Y      : {m['maxdev_x']:.2f} / {m['maxdev_y']:.2f}")
    print(f"Размах по X / по Y        : {m['range_x']:.2f} / {m['range_y']:.2f}")
    print(f"Сред. скорость качания    : {m['vel']:.2f} ед/с")
    print(f"RMS отклонение            : {m['rms']:.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Живой сбор стабилограммы с платформы Mera")
    ap.add_argument("--device", help="путь к устройству (Linux: /dev/input/eventN, Windows: COM-порт или hid://vid:pid)")
    ap.add_argument("--axes", help="JSON-конфиг осей и калибровки")
    ap.add_argument("--rate", type=float, default=50, help="частота дискретизации, Гц")
    ap.add_argument("--duration", type=float, default=0, help="авто-стоп через N с (0=нет)")
    ap.add_argument("--out", default="stabilo.csv", help="выходной CSV (имя без пути -> каталог records/)")
    ap.add_argument("--plot", default="stabilo.png", help="выходной PNG (имя без пути -> каталог records/)")
    ap.add_argument("--no-plot-show", action="store_true", help="не показывать окно")
    ap.add_argument("command", nargs="?", default="live",
                    choices=["live", "list"], help="режим")
    args = ap.parse_args(argv)
    args.out = str(resolve_out_path(args.out))
    args.plot = str(resolve_out_path(args.plot))

    cfg = load_axes_config(args.axes)
    if args.device:
        cfg["device"] = args.device
    if args.rate:
        cfg["rate"] = args.rate

    if args.command == "list":
        d = discover_device()
        print("Найдено устройство:", d if d else "не найдено")
        return

    path = cfg["device"] or discover_device()
    if not path:
        if sys.platform == "win32":
            print("Устройство Mera не найдено. Укажите --device COM3 (или другой COM-порт)", file=sys.stderr)
        else:
            print("Устройство Mera не найдено. Укажите --device /dev/input/eventN", file=sys.stderr)
        sys.exit(1)

    # Verify it's not a touchpad
    if not is_likely_stabilo_platform(path):
        print(f"Предупреждение: {path} похож на тачпад, а не на стабилоплатформу.", file=sys.stderr)
        print("Используйте --device для явного указания устройства.", file=sys.stderr)

    try:
        dev = Device(path, max_x=cfg.get("max_x"), max_y=cfg.get("max_y"))
        dev.start()
    except OSError as e:
        print(f"Не удалось открыть {path}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Устройство: {path}")
    print(f"Калибровка осей ({cfg['calibrate']} с) — можно встать на платформу...")
    xcode, ycode = auto_detect_axes(dev, cfg["calibrate"])
    print(f"Оси: X=code {xcode}, Y=code {ycode}")

    samples = []
    start_time = time.time()
    last_sample = 0.0
    interval = 1.0 / cfg["rate"]
    stop = threading.Event()

    def sample_once():
        nonlocal last_sample
        now = time.time()
        if now - last_sample < interval:
            return
        last_sample = now
        xv = dev.get(xcode)
        yv = dev.get(ycode)
        xraw, yraw = xv[0], yv[0]
        if xraw is None or yraw is None:
            return
        last_ts = max(dev.get(xcode)[1], dev.get(ycode)[1])
        x = xraw * cfg["scale_x"] + cfg["offset_x"]
        y = yraw * cfg["scale_y"] + cfg["offset_y"]
        contact = 1.0 if (now - last_ts) < cfg["contact_timeout"] else 0.0
        samples.append((now - start_time, x, y, contact))

    def finalize():
        arr = np.array(samples, dtype=float) if samples else np.empty((0, 4))
        m = compute_metrics(arr)
        csv_path = _build_csv_path(args.out, t_start=start_time)
        save_csv(arr, csv_path, t_start=start_time, t_end=time.time())
        fig = make_figure(arr, m)
        if args.plot:
            from pathlib import Path as _Path
            _Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.plot, dpi=120)
            plt.close(fig)
        print_summary(m)
        print(f"Сохранён CSV : {csv_path}")
        print(f"Сохранён PNG : {args.plot}")
        if m:
            print(f"Длительность : {m['T']:.2f} с, отсчётов: {m['n']}")

    interactive = (os.environ.get("DISPLAY") or sys.platform == "win32") and not args.no_plot_show

    if not interactive:
        print("Headless-режим: запись до Ctrl-C или --duration...")
        try:
            while not stop.is_set():
                sample_once()
                now = time.time()
                if args.duration and (now - start_time) > args.duration:
                    break
                time.sleep(interval / 2)
        except KeyboardInterrupt:
            pass
        dev.stop_now()
        finalize()
        return

    # ИСПРАВЛЕНО: используем GridSpec с выделенной осью для colorbar
    fig = plt.figure(figsize=(13, 5))
    gs = GridSpec(1, 3, width_ratios=[1, 0.05, 1], wspace=0.4)
    ax1 = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])  # Постоянная ось для colorbar
    ax2 = fig.add_subplot(gs[0, 2])

    def update(frame):
        sample_once()
        if not samples:
            return
        arr = np.array(samples)
        t, x, y, c = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        sel = c > 0.5

        # Очищаем все оси
        ax1.clear()
        ax2.clear()
        cax.clear()  # ИСПРАВЛЕНО: просто очищаем ось colorbar

        if sel.any():
            ax1.plot(x[sel], y[sel], "-", lw=0.8, color="gray", alpha=0.5)
            sc = ax1.scatter(x[sel], y[sel], c=t[sel], cmap="viridis", s=8)
            ax1.plot(x[sel][0], y[sel][0], "go", label="старт")
            ax1.plot(x[sel][-1], y[sel][-1], "ro", label="финиш")
            # ИСПРАВЛЕНО: используем cax вместо динамического создания
            fig.colorbar(sc, cax=cax, label="время, с")

        ax1.set_aspect("equal")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_title("Траектория ЦД (стабилограмма)")
        ax1.legend(loc="best")

        ax2.plot(t, x, "-", label="X")
        ax2.plot(t, y, "-", label="Y")
        if len(x):
            ax2.plot(t, c * (np.nanmax(np.abs(x)) + 1), "k", alpha=0.3, label="контакт")
        ax2.set_xlabel("время, с")
        ax2.set_ylabel("значение")
        ax2.set_title("Временные ряды")
        ax2.legend(loc="best")

        if args.duration and (time.time() - start_time) > args.duration:
            stop.set()
            plt.close(fig)

    anim = FuncAnimation(fig, update, interval=1000 / cfg["rate"], cache_frame_data=False)

    def on_key(event):
        if event.key == "q":
            stop.set()
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    print("Нажмите 'q' для остановки и сохранения, или закройте окно.")
    try:
        plt.show()
    except KeyboardInterrupt:
        stop.set()
    dev.stop_now()
    finalize()


if __name__ == "__main__":
    main()