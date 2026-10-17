#!/usr/bin/env python3
"""Стабилограмма MERA с Dear PyGui — только UI-фронтенд.

Ядро (устройство, семплирование, метрики, CSV) — в modules.core.
"""

import argparse
import contextlib
import os
import subprocess
import sys
import threading
import time

import numpy as np
import dearpygui.dearpygui as dpg

from modules.core import (
    DataBuffer,
    Device,
    auto_detect_axes,
    build_csv_path,
    compute_metrics,
    discover_device,
    is_likely_stabilo_platform,
    load_config,
    print_summary,
    save_csv,
)
from modules.config import resolve_out_path

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
        out_path = build_csv_path(self._out_path, meta=self.meta, t_start=t_start)
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
    out_path = build_csv_path(out_path, meta=meta, t_start=t_start)
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
    ap.add_argument("--no-filter", action="store_true",
                    help="отключить медианный фильтр и отсечение выбросов")
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
    if args.no_filter:
        cfg["median_window"] = 1
        cfg["outlier_sigma"] = 0

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
