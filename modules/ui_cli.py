#!/usr/bin/env python3
"""Запись стабилограммы MERA: консоль/matplotlib + headless-режим.

Ядро (устройство, семплирование, метрики, CSV) — в modules.core,
здесь только live-график matplotlib и точка входа main().
"""
import argparse
import os
import sys
import threading
import time

import numpy as np
import matplotlib

if os.environ.get("DISPLAY") or sys.platform == "win32":
    matplotlib.use("TkAgg")
else:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

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


def main(argv=None):
    ap = argparse.ArgumentParser(description="Живой сбор стабилограммы с платформы Mera")
    ap.add_argument("--device", help="путь к устройству (Linux: /dev/input/eventN, Windows: COM-порт или hid://vid:pid)")
    ap.add_argument("--axes", help="JSON-конфиг осей и калибровки")
    ap.add_argument("--rate", type=float, default=50, help="частота дискретизации, Гц")
    ap.add_argument("--duration", type=float, default=0, help="авто-стоп через N с (0=нет)")
    ap.add_argument("--out", default="stabilo.csv", help="выходной CSV (имя без пути -> каталог records/)")
    ap.add_argument("--plot", default="stabilo.png", help="выходной PNG (имя без пути -> каталог records/)")
    ap.add_argument("--no-plot-show", action="store_true", help="не показывать окно")
    ap.add_argument("--no-filter", action="store_true",
                    help="отключить медианный фильтр и отсечение выбросов")
    ap.add_argument("--id", default="", help="ID пациента")
    ap.add_argument("--gender", default="", help="пол (M/F)")
    ap.add_argument("--age", type=int, default=0, help="возраст")
    ap.add_argument("command", nargs="?", default="live",
                    choices=["live", "list"], help="режим")
    args = ap.parse_args(argv)
    args.out = str(resolve_out_path(args.out))
    args.plot = str(resolve_out_path(args.plot))

    cfg = load_config(args.axes)
    if args.device:
        cfg["device"] = args.device
    if args.rate:
        cfg["rate"] = args.rate
    if args.no_filter:
        cfg["median_window"] = 1
        cfg["outlier_sigma"] = 0

    if args.command == "list":
        d = discover_device()
        print("Найдено устройство:", d if d else "не найдено")
        return

    path = cfg["device"] or discover_device()
    if not path:
        if sys.platform == "win32":
            print("Устройство MERA не найдено. Укажите --device COM3 (или другой COM-порт)", file=sys.stderr)
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

    meta = {"id": args.id, "gender": args.gender, "age": args.age}
    buf = DataBuffer(cfg, dev, xcode, ycode)
    stop = threading.Event()

    def finalize():
        arr = buf.to_array()
        m = compute_metrics(arr)
        t_start = buf._start
        t_end = time.time()
        csv_path = build_csv_path(args.out, meta=meta, t_start=t_start)
        save_csv(arr, csv_path, meta=meta, t_start=t_start, t_end=t_end)
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
        sample_thread = threading.Thread(target=buf.sample_loop, daemon=True)
        sample_thread.start()
        try:
            while buf.running:
                if args.duration and buf.elapsed > args.duration:
                    break
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        buf.stop()
        dev.stop_now()
        finalize()
        dev.close()
        return

    fig = plt.figure(figsize=(13, 5))
    gs = GridSpec(1, 3, width_ratios=[1, 0.05, 1], wspace=0.4)
    ax1 = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])  # Постоянная ось для colorbar
    ax2 = fig.add_subplot(gs[0, 2])

    def update(frame):
        buf.sample_once()
        arr = buf.snapshot()
        if not arr.size:
            return
        t, x, y, c = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        sel = c > 0.5

        # Очищаем все оси
        ax1.clear()
        ax2.clear()
        cax.clear()

        if sel.any():
            ax1.plot(x[sel], y[sel], "-", lw=0.8, color="gray", alpha=0.5)
            sc = ax1.scatter(x[sel], y[sel], c=t[sel], cmap="viridis", s=8)
            ax1.plot(x[sel][0], y[sel][0], "go", label="старт")
            ax1.plot(x[sel][-1], y[sel][-1], "ro", label="финиш")
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

        if args.duration and buf.elapsed > args.duration:
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
    buf.stop()
    dev.stop_now()
    finalize()
    dev.close()


if __name__ == "__main__":
    main()
