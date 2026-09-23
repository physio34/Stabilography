"""
Расширенный анализатор стабилограмм для постурографии.

Реализует:
- Итоговый лист с 4 панелями: траектория, X(t), Y(t), гистограммы X и Y
- Эллипс доверия 95% на графике траектории
- Спектральный анализ FFT с выделением значимых диапазонов
- Стабилограф Ромберга (коэффициент Ромберга)
- Нормограмма (возрастно-половые нормы)
- Экспорт интерактивного HTML-отчёта (Plotly, встроенный)
"""

import logging
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

CHI_SQUARE_95 = 2.4477
TREMOR_POSTURAL_RANGE = (0.1, 3.0)
TREMOR_PARKINSONIAN_RANGE = (3.0, 12.0)
DEFAULT_ARTIFACT_THRESHOLD = 10.0
DEFAULT_ARTIFACT_WINDOW = 10
HISTOGRAM_BINS = 50


def _smart_round(value: float) -> float:
    """Округляет до 1 знака; если результат 0.0 — до первого значащего (не более 4 знаков)."""
    if value == 0:
        return 0.0
    r1 = round(value, 1)
    if r1 != 0.0:
        return r1
    abs_val = abs(value)
    decimals = min(4, max(1, int(np.ceil(-np.log10(abs_val)))))
    return round(value, decimals)

DEFAULT_NORMS = {
    "male": {
        "20-29": {"rms_x": 2.5, "rms_y": 3.0, "path": 80, "ellipse_area": 250},
        "30-39": {"rms_x": 2.8, "rms_y": 3.2, "path": 90, "ellipse_area": 300},
        "40-49": {"rms_x": 3.2, "rms_y": 3.8, "path": 100, "ellipse_area": 350},
        "50-59": {"rms_x": 3.8, "rms_y": 4.5, "path": 120, "ellipse_area": 450},
        "60-69": {"rms_x": 4.5, "rms_y": 5.2, "path": 140, "ellipse_area": 550},
        "70-79": {"rms_x": 5.5, "rms_y": 6.5, "path": 170, "ellipse_area": 700},
        "80+":   {"rms_x": 7.0, "rms_y": 8.0, "path": 200, "ellipse_area": 900},
    },
    "female": {
        "20-29": {"rms_x": 2.2, "rms_y": 2.8, "path": 70, "ellipse_area": 200},
        "30-39": {"rms_x": 2.5, "rms_y": 3.0, "path": 80, "ellipse_area": 250},
        "40-49": {"rms_x": 3.0, "rms_y": 3.5, "path": 95, "ellipse_area": 320},
        "50-59": {"rms_x": 3.5, "rms_y": 4.2, "path": 110, "ellipse_area": 400},
        "60-69": {"rms_x": 4.2, "rms_y": 5.0, "path": 130, "ellipse_area": 500},
        "70-79": {"rms_x": 5.2, "rms_y": 6.0, "path": 160, "ellipse_area": 650},
        "80+":   {"rms_x": 6.5, "rms_y": 7.5, "path": 190, "ellipse_area": 850},
    }
}


def load_stabilogram(file_path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Загружает данные стабилограммы из CSV или XLSX файла.
    Поддерживает CSV с метаданными в начале (ID, Gender, Age и т.д.).
    Возвращает (DataFrame, metadata_dict).
    """
    logger.info(f"Загрузка данных из: {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    metadata: Dict[str, str] = {}

    if file_path.suffix.lower() == '.csv':
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        header_idx = 0
        for i, line in enumerate(lines):
            cols = [c.strip().lstrip('#').lower() for c in line.split(',')]
            if 'x' in cols and 'y' in cols:
                header_idx = i
                break
            parts = [p.strip().lstrip('#') for p in line.strip().split(',')]
            if len(parts) == 2:
                metadata[parts[0].lower()] = parts[1]

        from io import StringIO
        data_lines = [lines[header_idx].lstrip('#')] + lines[header_idx + 1:]
        data_text = ''.join(data_lines)
        df = pd.read_csv(StringIO(data_text))

    elif file_path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Неподдерживаемый формат файла. Используйте CSV или XLSX.")

    required_cols = {'X', 'Y'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Данные должны содержать колонки: {required_cols}. Найдены: {df.columns.tolist()}")

    df = df.dropna(subset=['X', 'Y']).reset_index(drop=True)

    if 'Time' not in df.columns:
        logger.warning("Колонка 'Time' не найдена. Создана искусственная ось времени (50 Гц).")
        df['Time'] = np.arange(len(df)) / 50.0

    if metadata:
        logger.info(f"Метаданные из файла: {metadata}")

    return df, metadata


def reject_artifacts(df: pd.DataFrame, threshold: float = DEFAULT_ARTIFACT_THRESHOLD,
                     window: int = DEFAULT_ARTIFACT_WINDOW):
    """
    Удаляет артефакты — резкие скачки показаний, превышающие порог.
    Возвращает (df_clean, artifact_mask) — очищенный DataFrame и булеву маску артефактов.
    """
    logger.info(f"Пропуск артефактов (порог: {threshold} мм, окно: {window})...")
    df = df.copy()
    artifact_mask = np.zeros(len(df), dtype=bool)

    for col in ['X', 'Y']:
        signal = df[col].values
        diff = np.abs(np.diff(signal))

        bad_idx = np.where(diff > threshold)[0]

        if window > 1:
            expanded = set()
            for idx in bad_idx:
                for w in range(-window, window + 1):
                    pos = idx + w
                    if 0 <= pos < len(signal):
                        expanded.add(pos)
            bad_idx = np.array(sorted(expanded))
        else:
            bad_idx_all = set(bad_idx)
            bad_idx_all.update(bad_idx + 1)
            bad_idx = np.array(sorted(bad_idx_all))

        if len(bad_idx) > 0:
            artifact_mask[bad_idx] = True
            good_mask = np.ones(len(signal), dtype=bool)
            good_mask[bad_idx] = False
            good_idx = np.where(good_mask)[0]
            if len(good_idx) >= 2:
                df.loc[~good_mask, col] = np.interp(
                    bad_idx, good_idx, signal[good_idx]
                )

    n_artifacts = np.sum(artifact_mask)
    logger.info(f"Найдено и исправлено артефактов: {n_artifacts}")
    return df, artifact_mask


def compute_psd(signal: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Вычисляет спектральную плотность мощности (PSD) для сигнала."""
    signal = signal - np.mean(signal)
    n = len(signal)
    if n < 2:
        return np.array([0]), np.array([0])

    window = np.hanning(n)
    signal_windowed = signal * window

    fft_vals = np.fft.rfft(signal_windowed)
    power_spectrum = (np.abs(fft_vals) ** 2) / (fs * np.sum(window ** 2))
    freqs = np.fft.rfftfreq(n, d=1 / fs)

    if len(freqs) > 1:
        freqs = freqs[1:]
        power_spectrum = power_spectrum[1:]

    return freqs, power_spectrum


def calculate_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Рассчитывает основные стабилометрические метрики."""
    logger.info("Расчет стабилометрических метрик...")

    x = df['X'].values
    y = df['Y'].values
    time = df['Time'].values

    time_diffs = np.diff(time)
    fs = 1.0 / np.mean(time_diffs) if len(time_diffs) > 0 and np.mean(time_diffs) > 0 else 50.0

    total_time = time[-1] - time[0]
    n_samples = len(x)

    dx = np.diff(x)
    dy = np.diff(y)
    path_length = np.sum(np.sqrt(dx ** 2 + dy ** 2))
    mean_velocity = path_length / total_time if total_time > 0 else 0.0

    rms_x = np.sqrt(np.mean(x ** 2 - np.mean(x) ** 2)) if n_samples > 1 else 0.0
    rms_y = np.sqrt(np.mean(y ** 2 - np.mean(y) ** 2)) if n_samples > 1 else 0.0

    range_x = np.max(x) - np.min(x)
    range_y = np.max(y) - np.min(y)

    x_c = x - np.mean(x)
    y_c = y - np.mean(y)
    cov_matrix = np.cov(x_c, y_c)

    try:
        eigenvalues, _ = np.linalg.eigh(cov_matrix)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        a = np.sqrt(eigenvalues[0]) * CHI_SQUARE_95
        b = np.sqrt(eigenvalues[1]) * CHI_SQUARE_95
        ellipse_area = np.pi * a * b
    except np.linalg.LinAlgError:
        logger.error("Ошибка вычисления ковариационной матрицы.")
        ellipse_area = 0.0

    freqs_x, power_x = compute_psd(x, fs)
    freqs_y, power_y = compute_psd(y, fs)

    postural_x = np.sum(power_x[(freqs_x >= TREMOR_POSTURAL_RANGE[0]) & (freqs_x <= TREMOR_POSTURAL_RANGE[1])]) if len(freqs_x) > 0 else 0.0
    postural_y = np.sum(power_y[(freqs_y >= TREMOR_POSTURAL_RANGE[0]) & (freqs_y <= TREMOR_POSTURAL_RANGE[1])]) if len(freqs_y) > 0 else 0.0
    parkinsonian_x = np.sum(power_x[(freqs_x >= TREMOR_PARKINSONIAN_RANGE[0]) & (freqs_x <= TREMOR_PARKINSONIAN_RANGE[1])]) if len(freqs_x) > 0 else 0.0
    parkinsonian_y = np.sum(power_y[(freqs_y >= TREMOR_PARKINSONIAN_RANGE[0]) & (freqs_y <= TREMOR_PARKINSONIAN_RANGE[1])]) if len(freqs_y) > 0 else 0.0

    peak_freq_x = freqs_x[np.argmax(power_x)] if len(freqs_x) > 0 and np.max(power_x) > 0 else 0.0
    peak_freq_y = freqs_y[np.argmax(power_y)] if len(freqs_y) > 0 and np.max(power_y) > 0 else 0.0

    metrics = {
        'Общее время (с)': _smart_round(total_time),
        'Частота дискретизации (Гц)': _smart_round(fs),
        'Количество точек': n_samples,
        'Длина пути (мм)': _smart_round(path_length),
        'Средняя скорость (мм/с)': _smart_round(mean_velocity),
        'СКО по оси X (мм)': _smart_round(rms_x),
        'СКО по оси Y (мм)': _smart_round(rms_y),
        'Размах по оси X (мм)': _smart_round(range_x),
        'Размах по оси Y (мм)': _smart_round(range_y),
        'Площадь 95% эллипса (мм²)': _smart_round(ellipse_area),
        'Мощность постур. тремора X (0.1-3 Гц)': _smart_round(postural_x),
        'Мощность постур. тремора Y (0.1-3 Гц)': _smart_round(postural_y),
        'Мощность паркинс. тремора X (3-12 Гц)': _smart_round(parkinsonian_x),
        'Мощность паркинс. тремора Y (3-12 Гц)': _smart_round(parkinsonian_y),
        'Пиковая частота X (Гц)': _smart_round(peak_freq_x),
        'Пиковая частота Y (Гц)': _smart_round(peak_freq_y),
    }

    return metrics


def compute_romberg_coefficient(df_open: pd.DataFrame, df_closed: pd.DataFrame) -> Dict[str, float]:
    """Вычисляет коэффициент Ромберга L_closed / L_open * 100%."""
    logger.info("Вычисление коэффициента Ромберга...")

    def _path_length(dframe):
        x = dframe['X'].values
        y = dframe['Y'].values
        dx = np.diff(x)
        dy = np.diff(y)
        return np.sum(np.sqrt(dx ** 2 + dy ** 2))

    l_open = _path_length(df_open)
    l_closed = _path_length(df_closed)

    romberg_coeff = (l_closed / l_open * 100.0) if l_open > 0 else 0.0

    rms_open_x = np.sqrt(np.mean((df_open['X'].values - np.mean(df_open['X'].values)) ** 2))
    rms_open_y = np.sqrt(np.mean((df_open['Y'].values - np.mean(df_open['Y'].values)) ** 2))
    rms_closed_x = np.sqrt(np.mean((df_closed['X'].values - np.mean(df_closed['X'].values)) ** 2))
    rms_closed_y = np.sqrt(np.mean((df_closed['Y'].values - np.mean(df_closed['Y'].values)) ** 2))

    return {
        'Длина пути (открытые глаза) (мм)': l_open,
        'Длина пути (закрытые глаза) (мм)': l_closed,
        'Коэффициент Ромберга (%)': romberg_coeff,
        'СКО X открытые (мм)': rms_open_x,
        'СКО Y открытые (мм)': rms_open_y,
        'СКО X закрытые (мм)': rms_closed_x,
        'СКО Y закрытые (мм)': rms_closed_y,
    }


def _get_age_group(age: int) -> str:
    """Определяет возрастную группу."""
    if age < 30:
        return "20-29"
    elif age < 40:
        return "30-39"
    elif age < 50:
        return "40-49"
    elif age < 60:
        return "50-59"
    elif age < 70:
        return "60-69"
    elif age < 80:
        return "70-79"
    else:
        return "80+"


def load_norms(norms_path: Optional[Path] = None) -> Dict:
    """Загружает нормы из JSON файла или возвращает дефолтные."""
    candidates = []
    if norms_path:
        candidates.append(Path(norms_path))
    candidates.append(Path(__file__).parent / "norms.json")
    candidates.append(Path(__file__).parent.parent / "norms.json")

    for p in candidates:
        if p.exists():
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"Нормы загружены из: {p}")
                    return data
            except Exception as e:
                logger.warning(f"Ошибка загрузки норм из {p}: {e}")

    logger.info("Используются встроенные дефолтные нормы.")
    return DEFAULT_NORMS


def check_norms(metrics: Dict[str, float], age: int, gender: str,
                norms: Dict) -> Dict[str, Dict]:
    """Проверяет отклонения от норм по возрастно-половым критериям."""
    gender_key = gender.lower() if gender.lower() in norms else "male"
    age_group = _get_age_group(age)
    age_norms = norms.get(gender_key, norms.get("male", {})).get(age_group, {})

    results = {}
    metric_map = {
        'СКО по оси X (мм)': 'rms_x',
        'СКО по оси Y (мм)': 'rms_y',
        'Длина пути (мм)': 'path',
        'Площадь 95% эллипса (мм²)': 'ellipse_area',
    }

    for metric_name, norm_key in metric_map.items():
        if metric_name in metrics and norm_key in age_norms:
            value = metrics[metric_name]
            norm_value = age_norms[norm_key]
            ratio = value / norm_value if norm_value > 0 else 0.0
            status = "normal"
            if ratio > 1.5:
                status = "critical"
            elif ratio > 1.2:
                status = "warning"
            results[metric_name] = {
                'value': value,
                'norm': norm_value,
                'ratio': ratio,
                'status': status,
            }

    return results


def _ellipse_points(cov_matrix: np.ndarray, center: np.ndarray,
                    chi2: float = CHI_SQUARE_95, n_pts: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Вычисляет точки эллипса доверия по ковариационной матрице."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    a = np.sqrt(max(eigenvalues[0], 0)) * chi2
    b = np.sqrt(max(eigenvalues[1], 0)) * chi2
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

    t = np.linspace(0, 2 * np.pi, n_pts)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    pts = R @ np.array([a * np.cos(t), b * np.sin(t)])
    return pts[0] + center[0], pts[1] + center[1]


def generate_extended_html(
    df: pd.DataFrame,
    metrics: Dict[str, float],
    romberg_metrics: Optional[Dict[str, float]] = None,
    norm_results: Optional[Dict[str, Dict]] = None,
    output_path: Path = Path("extended_report.html"),
    patient_info: Optional[Dict] = None,
    artifact_mask: Optional[np.ndarray] = None,
) -> None:
    """Генерирует расширенный интерактивный HTML-отчёт (Plotly, встроенный)."""
    logger.info(f"Генерация расширенного HTML-отчёта: {output_path}")

    x = df['X'].values
    y = df['Y'].values
    time = df['Time'].values
    x_c = x - np.mean(x)
    y_c = y - np.mean(y)
    cov_matrix = np.cov(x_c, y_c)
    ell_x, ell_y = _ellipse_points(cov_matrix, np.array([np.mean(x), np.mean(y)]))

    time_diffs = np.diff(time)
    fs = 1.0 / np.mean(time_diffs) if len(time_diffs) > 0 and np.mean(time_diffs) > 0 else 50.0
    freqs_x, power_x = compute_psd(x, fs)
    freqs_y, power_y = compute_psd(y, fs)

    psd_all = np.concatenate([power_x, power_y])
    psd_all = psd_all[psd_all > 0]
    if len(psd_all) > 0:
        psd_ymin = 10 ** np.floor(np.log10(psd_all.min()))
        psd_ymax = 10 ** np.ceil(np.log10(psd_all.max()))
    else:
        psd_ymin, psd_ymax = 1e-10, 1.0

    fig = make_subplots(
        rows=4, cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        subplot_titles=(
            "Статокинезиограмма (X vs Y) + 95% эллипс",
            "Отклонение X(t)",
            "Отклонение Y(t)",
            "Гистограмма X",
            "Гистограмма Y",
            "Спектр мощности X (PSD)",
            "Спектр мощности Y (PSD)",
        ),
        vertical_spacing=0.06,
        horizontal_spacing=0.08,
    )

    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines+markers', name='Траектория',
        line=dict(width=1, color='steelblue'), marker=dict(size=2, color='steelblue'),
        opacity=0.7,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=ell_x, y=ell_y, mode='lines', name='95% Эллипс',
        line=dict(color='red', width=2.5, dash='dash'),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[np.mean(x)], y=[np.mean(y)], mode='markers', name='Центр',
        marker=dict(size=10, color='red', symbol='cross'),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=time, y=x, mode='lines', name='X(t)',
        line=dict(color='royalblue', width=1),
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=time, y=y, mode='lines', name='Y(t)',
        line=dict(color='darkorange', width=1),
    ), row=2, col=2)

    fig.add_trace(go.Histogram(
        x=x, name='Гистограмма X', marker_color='royalblue', opacity=0.7,
        nbinsx=HISTOGRAM_BINS,
    ), row=3, col=1)

    fig.add_trace(go.Histogram(
        x=y, name='Гистограмма Y', marker_color='darkorange', opacity=0.7,
        nbinsx=HISTOGRAM_BINS,
    ), row=3, col=2)

    fig.add_trace(go.Scatter(
        x=freqs_x, y=power_x, mode='lines', name='PSD X',
        line=dict(color='royalblue', width=1.5),
    ), row=4, col=1)

    fig.add_trace(go.Scatter(
        x=freqs_y, y=power_y, mode='lines', name='PSD Y',
        line=dict(color='darkorange', width=1.5),
    ), row=4, col=2)

    if artifact_mask is not None and np.any(artifact_mask):
        ax = x[artifact_mask]
        ay = y[artifact_mask]
        at = time[artifact_mask]
        fig.add_trace(go.Scatter(
            x=ax, y=ay, mode='markers', name='Артефакты',
            marker=dict(size=7, color='red', symbol='x', line=dict(width=1.5)),
            showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=at, y=ax, mode='markers', name='Артефакты X',
            marker=dict(size=5, color='red', symbol='x', line=dict(width=1)),
            showlegend=False,
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=at, y=ay, mode='markers', name='Артефакты Y',
            marker=dict(size=5, color='red', symbol='x', line=dict(width=1)),
            showlegend=False,
        ), row=2, col=2)

    fig.add_vrect(
        x0=TREMOR_POSTURAL_RANGE[0], x1=TREMOR_POSTURAL_RANGE[1],
        fillcolor="green", opacity=0.1, line_width=0,
        annotation_text="0.1–3 Гц<br>(постуральный тремор)",
        annotation_position="top left",
        row=4, col=1,
    )
    fig.add_vrect(
        x0=TREMOR_PARKINSONIAN_RANGE[0], x1=TREMOR_PARKINSONIAN_RANGE[1],
        fillcolor="red", opacity=0.1, line_width=0,
        annotation_text="3–12 Гц<br>(паркинсоноподобный тремор)",
        annotation_position="top right",
        row=4, col=1,
    )

    fig.add_vrect(
        x0=TREMOR_POSTURAL_RANGE[0], x1=TREMOR_POSTURAL_RANGE[1],
        fillcolor="green", opacity=0.1, line_width=0,
        annotation_text="0.1–3 Гц<br>(постуральный тремор)",
        annotation_position="top left",
        row=4, col=2,
    )
    fig.add_vrect(
        x0=TREMOR_PARKINSONIAN_RANGE[0], x1=TREMOR_PARKINSONIAN_RANGE[1],
        fillcolor="red", opacity=0.1, line_width=0,
        annotation_text="3–12 Гц<br>(паркинсоноподобный тремор)",
        annotation_position="top right",
        row=4, col=2,
    )

    fig.update_xaxes(title_text="X (мм)", row=1, col=1)
    fig.update_yaxes(title_text="Y (мм)", row=1, col=1, scaleanchor="x", scaleratio=1)
    fig.update_xaxes(title_text="Время (с)", row=2, col=1)
    fig.update_yaxes(title_text="X (мм)", row=2, col=1)
    fig.update_xaxes(title_text="Время (с)", row=2, col=2)
    fig.update_yaxes(title_text="Y (мм)", row=2, col=2)
    fig.update_xaxes(title_text="X (мм)", row=3, col=1)
    fig.update_yaxes(title_text="Частота встречаемости", row=3, col=1)
    fig.update_xaxes(title_text="Y (мм)", row=3, col=2)
    fig.update_yaxes(title_text="Частота встречаемости", row=3, col=2)
    fig.update_xaxes(title_text="Частота (Гц)", row=4, col=1, type="log",
                     range=[np.log10(0.05), np.log10(fs / 2)])
    fig.update_yaxes(title_text="Мощность (мм²/Гц)", row=4, col=1, type="log",
                     range=[np.log10(psd_ymin), np.log10(psd_ymax)])
    fig.update_xaxes(title_text="Частота (Гц)", row=4, col=2, type="log",
                     range=[np.log10(0.05), np.log10(fs / 2)])
    fig.update_yaxes(title_text="Мощность (мм²/Гц)", row=4, col=2, type="log",
                     matches="y6", side="right")

    fig.update_layout(
        title_text="Расширенный отчёт постурографии",
        showlegend=False,
        height=2000,
        width=1100,
        template="plotly_white",
    )

    html_parts = []
    html_parts.append("<html><head><meta charset='utf-8'>")
    html_parts.append("<title>Расширенный отчёт постурографии</title>")
    html_parts.append("<style>")
    html_parts.append("body { font-family: Arial, sans-serif; margin: 20px; background: #f9f9f9; }")
    html_parts.append("h1 { color: #333; text-align: center; }")
    html_parts.append("h2 { color: #555; border-bottom: 2px solid #4a90d9; padding-bottom: 5px; }")
    html_parts.append("table { border-collapse: collapse; width: 100%; margin: 15px 0; }")
    html_parts.append("th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }")
    html_parts.append("th { background-color: #4a90d9; color: white; }")
    html_parts.append("tr:nth-child(even) { background-color: #f2f2f2; }")
    html_parts.append(".normal { background-color: #d4edda; }")
    html_parts.append(".warning { background-color: #fff3cd; }")
    html_parts.append(".critical { background-color: #f8d7da; }")
    html_parts.append(".patient-info { background: #e8f4f8; padding: 15px; border-radius: 8px; margin: 15px 0; }")
    html_parts.append("</style></head><body>")

    html_parts.append("<h1>Расширенный отчёт постурографии</h1>")

    if patient_info:
        html_parts.append("<div class='patient-info'>")
        html_parts.append("<h2>Информация о пациенте</h2><ul>")
        for key, val in patient_info.items():
            html_parts.append(f"<li><strong>{key}:</strong> {val}</li>")
        html_parts.append("</ul></div>")

    plotly_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
    html_parts.append(plotly_html)

    html_parts.append("<h2>Основные метрики</h2>")
    html_parts.append("<table><tr><th>Показатель</th><th>Значение</th></tr>")
    for key, val in metrics.items():
        val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
        html_parts.append(f"<tr><td>{key}</td><td>{val_str}</td></tr>")
    html_parts.append("</table>")

    if romberg_metrics:
        html_parts.append("<h2>Тест Ромберга</h2>")

        l_open = romberg_metrics.get('Длина пути (открытые глаза) (мм)', 0)
        l_closed = romberg_metrics.get('Длина пути (закрытые глаза) (мм)', 0)
        romberg_coeff = romberg_metrics.get('Коэффициент Ромберга (%)', 0)

        fig_romberg = go.Figure()
        fig_romberg.add_trace(go.Bar(
            x=['Открытые глаза', 'Закрытые глаза'],
            y=[l_open, l_closed],
            marker_color=['steelblue', 'darkorange'],
            text=[f"{l_open:.1f} мм", f"{l_closed:.1f} мм"],
            textposition='auto',
        ))
        fig_romberg.update_layout(
            title=f"Тест Ромберга — Коэффициент: {romberg_coeff:.1f}%",
            yaxis_title="Длина пути (мм)",
            template="plotly_white",
            height=400,
            width=700,
        )

        romberg_status = "норма" if 80 <= romberg_coeff <= 120 else "патология"
        html_parts.append(f"<p>Коэффициент Ромберга: <strong>{romberg_coeff:.1f}%</strong> ({romberg_status})</p>")
        html_parts.append(fig_romberg.to_html(full_html=False, include_plotlyjs='cdn'))

        html_parts.append("<table><tr><th>Показатель</th><th>Значение</th></tr>")
        for key, val in romberg_metrics.items():
            val_str = f"{val:.2f}" if isinstance(val, float) else str(val)
            html_parts.append(f"<tr><td>{key}</td><td>{val_str}</td></tr>")
        html_parts.append("</table>")

    if norm_results:
        html_parts.append("<h2>Нормограмма</h2>")
        html_parts.append("<table><tr><th>Показатель</th><th>Значение</th><th>Норма</th><th>Отношение</th><th>Статус</th></tr>")
        for metric_name, info in norm_results.items():
            status = info['status']
            status_text = {"normal": "Норма", "warning": "Отклонение", "critical": "Критично"}.get(status, "?")
            row_class = status
            html_parts.append(
                f"<tr class='{row_class}'><td>{metric_name}</td>"
                f"<td>{info['value']:.2f}</td>"
                f"<td>{info['norm']:.2f}</td>"
                f"<td>{info['ratio']:.2f}</td>"
                f"<td>{status_text}</td></tr>"
            )
        html_parts.append("</table>")

    html_parts.append("</body></html>")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))

    logger.info(f"Отчёт сохранён: {output_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Расширенный анализ стабилограмм с клиническими заключениями."
    )
    parser.add_argument('input_file', type=str, nargs='?',
                        help='Путь к файлу стабилограммы (CSV/XLSX); без него — интерактивный выбор')
    parser.add_argument('-o', '--output_dir', type=str, default='.', help='Директория для отчётов')
    parser.add_argument('--dir', type=str, default=None,
                        help='Каталог для интерактивного выбора (по умолчанию records/)')
    parser.add_argument('--age', type=int, default=None, help='Возраст пациента (для нормограммы, необязательно)')
    parser.add_argument('--gender', type=str, default=None, choices=['male', 'female'],
                        help='Пол пациента (male/female, необязательно)')
    parser.add_argument('--norms', type=str, default=None,
                        help='Путь к JSON-файлу с нормами (по умолчанию — встроенные)')
    parser.add_argument('--romberg_open', type=str, default=None,
                        help='Файл стабилограммы с открытыми глазами (тест Ромберга)')
    parser.add_argument('--romberg_closed', type=str, default=None,
                        help='Файл стабилограммы с закрытыми глазами (тест Ромберга)')
    parser.add_argument('--patient_name', type=str, default='Не указано', help='Имя пациента')
    parser.add_argument('--artifact_threshold', type=float, default=None,
                        help='Порог пропуска артефактов в мм (по умолчанию 10.0; 0 — отключить)')

    args = parser.parse_args(argv)

    try:
        from modules.file_select import resolve_input_file
    except ImportError:
        from .file_select import resolve_input_file
    try:
        input_file = resolve_input_file(args.input_file, args.dir)
    except KeyboardInterrupt as e:
        logger.info(str(e))
        return
    input_path = Path(input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        df, file_metadata = load_stabilogram(input_path)
        artifact_mask = None

        if args.artifact_threshold is None or args.artifact_threshold > 0:
            thr = args.artifact_threshold if args.artifact_threshold else DEFAULT_ARTIFACT_THRESHOLD
            df, artifact_mask = reject_artifacts(df, threshold=thr)

        metrics = calculate_metrics(df)

        age = args.age
        gender = args.gender
        if age is None and 'age' in file_metadata:
            try:
                age = int(file_metadata['age'])
            except ValueError:
                pass
        if gender is None and 'gender' in file_metadata:
            g = file_metadata['gender'].strip().lower()
            if g in ('m', 'male', 'м', 'мужской'):
                gender = 'male'
            elif g in ('f', 'female', 'ж', 'женский'):
                gender = 'female'

        patient_info = {
            'Имя': args.patient_name,
            'Файл': input_path.name,
        }
        for k, v in file_metadata.items():
            if k not in ('gender', 'age'):
                patient_info[k] = v
        if age is not None:
            patient_info['Возраст'] = age
        if gender is not None:
            patient_info['Пол'] = 'Мужской' if gender == 'male' else 'Женский'

        romberg_metrics = None
        if args.romberg_open and args.romberg_closed:
            df_open, _ = load_stabilogram(Path(args.romberg_open))
            df_closed, _ = load_stabilogram(Path(args.romberg_closed))
            romberg_metrics = compute_romberg_coefficient(df_open, df_closed)
            patient_info['Тест Ромберга'] = 'Выполнен'

        norm_results = None
        if age is not None and gender is not None:
            norms_path = Path(args.norms) if args.norms else None
            norms = load_norms(norms_path)
            norm_results = check_norms(metrics, age, gender, norms)
        else:
            logger.info("Возраст/пол не указаны — нормограмма пропущена.")

        html_path = output_dir / f"{input_path.stem}_extended_report.html"
        generate_extended_html(
            df, metrics, romberg_metrics, norm_results,
            html_path, patient_info, artifact_mask,
        )

        logger.info("Анализ завершён успешно!")

    except Exception as e:
        logger.error(f"Ошибка: {e}", exc_info=True)


if __name__ == "__main__":
    main()
