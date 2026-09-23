"""Интерактивный выбор стабилограммы (логика перенесена из menu.py).

Поведение:
- ищет *.csv/*.xlsx/*.xls в заданном каталоге (по умолчанию records/);
- сортировка по умолчанию — сначала новые;
- пустой ввод (Enter) выбирает самый новый файл;
- поддерживаются команды: sort name|new|ext, /текст, ext xxx, c, q/0.
"""
import datetime
import os

from .config import RECORDS_DIR, SUPPORTED_STABILO_EXTS
from . import console


def find_files(directory=None, extensions=SUPPORTED_STABILO_EXTS):
    """Найти поддерживаемые файлы в каталоге (сначала новые)."""
    if directory is None:
        directory = str(RECORDS_DIR)
    result = []
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    for f in sorted(names):
        if f.lower().endswith(extensions):
            full = os.path.join(directory, f)
            try:
                mtime = os.stat(full).st_mtime
            except OSError:
                continue
            result.append(full)
    result.sort(key=lambda p: os.stat(p).st_mtime, reverse=True)
    return result


def interactive_select_file(directory=None, extensions=SUPPORTED_STABILO_EXTS):
    """Меню выбора файла. Возвращает путь или None.

    Пустой ввод (Enter) — самый новый файл (первый в списке).
    """
    if directory is None:
        directory = str(RECORDS_DIR)
    sort_mode = "new"  # name | new | ext — по умолчанию сначала новые
    query = ""
    ext_filter = ""
    min_size = 0  # стабилограммы маленькие, порог не нужен

    def collect():
        files = []
        try:
            names = sorted(os.listdir(directory))
        except OSError:
            print(console.c_red(f"Каталог недоступен: {directory}"))
            return []
        for name in names:
            if not name.lower().endswith(extensions):
                continue
            if query and query not in name.lower():
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext_filter and ext != ext_filter:
                continue
            full = os.path.join(directory, name)
            try:
                stat = os.stat(full)
            except OSError:
                continue
            if stat.st_size < min_size:
                continue
            files.append({"path": full, "name": name, "ext": ext,
                          "size": stat.st_size, "mtime": stat.st_mtime})
        if sort_mode == "new":
            files.sort(key=lambda f: f["mtime"], reverse=True)
        elif sort_mode == "ext":
            files.sort(key=lambda f: (f["ext"], f["name"].lower()))
        else:
            files.sort(key=lambda f: f["name"].lower())
        return files

    sort_names = {"name": "имени", "new": "новизне", "ext": "расширению"}
    while True:
        files = collect()
        if not files:
            print(console.c_yellow("Файлы не найдены (измените поиск/фильтр или каталог)."))
        else:
            print(console.c_bold(
                f"\nДоступные файлы в {directory} "
                f"(сортировка: {sort_names[sort_mode]}"
                + (f", поиск: '{query}'" if query else "")
                + (f", фильтр: '{ext_filter}'" if ext_filter else "")
                + "):"))
            for i, f in enumerate(files, 1):
                stamp = datetime.datetime.fromtimestamp(
                    f["mtime"]).strftime("%Y-%m-%d %H:%M")
                mark = console.c_green("  <-- самый новый (Enter)") if i == 1 else ""
                print(f"  {console.c_cyan(f'{i:3d}.')} "
                      f"{console.c_yellow(f['name'])}  "
                      f"({console.c_green(f'{f['size'] / 1024:.0f} КБ')}, "
                      f"{console.c_dim(stamp)}){mark}")
        print(f"  {console.c_dim('0 / Q. Выход')}   "
              f"{console.c_dim('sort name|new|ext, /текст, ext XXX, c — сброс, Enter — самый новый')}")

        try:
            choice = input("Номер, команда или Enter: ").strip()
        except EOFError:
            return None
        if not choice:
            return files[0]["path"] if files else None
        if choice == "0" or choice.lower() == "q":
            return None
        low = choice.lower()
        if low.startswith("sort"):
            parts = low.split()
            mode = parts[1] if len(parts) > 1 else ""
            names_map = {"name": "name", "new": "new", "ext": "ext",
                         "имени": "name", "дате": "new", "типу": "ext"}
            if mode in names_map:
                sort_mode = names_map[mode]
                continue
            print(console.c_red("Сортировка: sort name|new|ext"))
            continue
        if low.startswith("/"):
            query = low[1:].strip()
            continue
        if low.startswith("ext ") or low.startswith("f "):
            arg = low.split(None, 1)[1].strip().lstrip(".") \
                if len(low.split(None, 1)) > 1 else ""
            if not arg or arg == "*":
                ext_filter = ""
            else:
                ext_filter = "." + arg
                if ext_filter not in extensions:
                    print(console.c_red(
                        f"Неизвестное расширение. Доступны: {' '.join(extensions)}"))
                    ext_filter = ""
            continue
        if low == "c":
            query = ""
            ext_filter = ""
            continue
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                return files[idx]["path"]
            print(console.c_red(f"Введите число от 1 до {len(files)}."))
        except ValueError:
            print(console.c_red(
                "Некорректный ввод (номер, sort, /текст, ext, c, q, Enter)."))


def resolve_input_file(path_or_none, directory=None):
    """Вернуть путь к файлу: явный --file либо интерактивный выбор."""
    if path_or_none:
        if not os.path.isfile(path_or_none):
            raise FileNotFoundError(f"Файл не найден: {path_or_none}")
        return path_or_none
    if directory is None:
        directory = str(RECORDS_DIR)
    os.makedirs(directory, exist_ok=True)
    picked = interactive_select_file(directory)
    if picked is None:
        raise KeyboardInterrupt("Файл не выбран.")
    print(console.c_green(f"Выбран: {picked}"))
    return picked
