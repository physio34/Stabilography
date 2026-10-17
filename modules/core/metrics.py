"""Стабилографические метрики из массива N×4 [t, x, y, contact]."""
import numpy as np


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
