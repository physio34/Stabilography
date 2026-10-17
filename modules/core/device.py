"""Слой устройства: поиск MERA, чтение evdev/HID/serial, валидация, кольцевой буфер."""
import collections
import os
import re
import select
import struct
import sys
import threading
import time

import numpy as np

# ── evdev constants ────────────────────────────────────────────────────────────
EV_ABS = 3
EVENT_FMT = "llHHi"
EVENT_SIZE = struct.calcsize(EVENT_FMT)
ABSINFO_FMT = "iiiiii"
ABSINFO_SIZE = struct.calcsize(ABSINFO_FMT)


def _eviocgabs(fd: int, code: int) -> tuple:
    """Возвращает (value, min, max, fuzz, flat, resolution) оси evdev."""
    import fcntl
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
        import contextlib
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
        # Тачпады/мыши дают значения вне диапазона стабилоплатформы.
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
    """Определяет коды X/Y по диапазону осей и дисперсии за window_sec секунд."""
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
