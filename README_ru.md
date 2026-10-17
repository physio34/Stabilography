# adStabilo — запись и анализ стабилограмм

Программа для работы со стабилоплатформой **MERA**: живая запись стабилограммы
(траектории центра давления) и последующий анализ записей с интерактивными
HTML-отчётами (траектория + 95% эллипс, временные ряды, гистограммы, спектры PSD,
тест Ромберга, нормограмма).

## Структура проекта

```
adStabilo/
├── main.py              # Точка входа: цветное меню + CLI
├── modules/
│   ├── core/            # Общее ядро записи
│   │   ├── device.py    # Device, поиск MERA, валидация
│   │   ├── sampling.py  # DataBuffer, калибровка осей
│   │   ├── metrics.py   # compute_metrics, print_summary
│   │   ├── io_csv.py    # save_csv, build_csv_path
│   │   └── defaults.py  # load_config (единые дефолты)
│   ├── ui_cli.py        # Запись (консоль/matplotlib, headless)
│   ├── ui_gui.py        # Запись (GUI Dear PyGui)
│   ├── analyze.py       # Быстрый анализ  → <имя>_report.html
│   ├── extended.py      # Расширенный анализ → <имя>_extended_report.html
│   ├── file_select.py   # Интерактивный выбор файлов
│   ├── config.py        # Каталог records/ по умолчанию
│   ├── console.py       # ANSI-цвета меню
│   └── norms.json       # Возрастно-половые нормы
├── records/             # Записанные стабилограммы (CSV)
├── analyze.md           # Детали анализа и метрик
├── record.md            # Детали записи и алгоритма
└── README_ru.md         # Этот файл
```

## Требования

- Python 3.9+
- Запись: `numpy`, `matplotlib` (+ `dearpygui` для GUI, `pyserial`/`hid` на Windows)
- Анализ: `numpy`, `pandas`, `plotly`

```bash
pip install -r requirements.txt
```

## Использование

Интерактивное меню (цветное):

```bash
python main.py
```

```
=== adStabilo ===
  1. Запись (консоль/matplotlib)
  2. Запись (GUI Dear PyGui)
  3. Анализ расширенный (артефакты, Ромберг, нормы)
  4. Анализ быстрый
  5. Список устройств
  0 / Q. Выход
```

Команды (аргументы после команды передаются соответствующему модулю):

```bash
python main.py record [opts]            # запись, matplotlib/headless
python main.py gui [opts]               # запись, Dear PyGui
python main.py analyze [файл] [opts]    # расширенный анализ
python main.py quick [файл] [opts]      # быстрый анализ
python main.py list                     # поиск устройства MERA
python main.py -f ФАЙЛ [--dir КАТАЛОГ] [--quick] [opts анализа]
```

Примеры:

```bash
python main.py record --duration 30
python main.py analyze --dir records -o ./reports --age 30 --gender male
python main.py quick records/stabilo.csv -o ./reports
python main.py -f records/stabilo.csv --age 30 --gender male
```

Прямой запуск модулей тоже работает:

```bash
python -m modules.ui_cli list
python -m modules.extended --help
python -m modules.analyze --help
```

## Записи: каталог records/

Выходной CSV без пути (`--out stabilo.csv`) сохраняется в `records/`
(см. `modules/config.py`). Имя файла дополняется меткой времени и данными
пациента: `stabilo_YYYYMMDD_HHMMSS[_ID][_пол][_возраст].csv`.

## Анализ: интерактивный выбор файла

Без указания файла открывается цветное меню выбора из каталога
(`--dir` или `records/` по умолчанию):

- сортировка по умолчанию — сначала новые;
- пустой ввод (Enter) — самый новый файл;
- команды: `sort name|new|ext`, `/текст` (поиск), `ext xxx` (фильтр),
  `c` (сброс), `q`/`0` (выход).

## Отчёты

| Режим    | Команда              | Файл отчёта                |
|----------|----------------------|----------------------------|
| Быстрый  | `quick` (п. 4 меню)  | `<имя>_report.html`        |
| Расширенный | `analyze` (п. 3 меню) | `<имя>_extended_report.html` |

Расширенный анализ дополнительно чистит артефакты
(`--artifact_threshold`, по умолчанию 10 мм; `0` — отключить),
подхватывает возраст/пол из метаданных CSV и строит нормограмму
при `--age` + `--gender`. Тест Ромберга — через
`--romberg_open` + `--romberg_closed`.

## Формат входных данных

CSV/XLSX с колонками `X`, `Y` (мм) и опционально `Time` (с).
Строки метаданных в начале CSV (`ID`, `Gender`, `Age`, `#...`)
пропускаются автоматически. Без `Time` создаётся ось 50 Гц.

## Лицензия

MIT
