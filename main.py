#!/usr/bin/env python3
"""adStabilo — точка входа: запись и анализ стабилограмм.

Использование:
    python main.py                          # интерактивное меню
    python main.py record [opts]            # запись (консоль/matplotlib, modules/ui_cli.py)
    python main.py gui [opts]               # запись (Dear PyGui, modules/ui_gui.py)
    python main.py analyze [file] [opts]    # расширенный анализ (modules/extended.py)
    python main.py quick [file] [opts]      # быстрый анализ (modules/analyze.py)
    python main.py list                     # поиск устройства MERA
    python main.py -f FILE [--dir DIR] [--quick] [opts анализа]
                                            # single-shot анализ файла (логика из menu.py)

Интерактивный выбор файла при анализе: заданный каталог (--dir) или
records/ по умолчанию; сортировка — сначала новые, Enter — самый новый.
Записи по умолчанию сохраняются в каталоге records/.
"""

import argparse
import sys

from modules.config import default_records_dir
from modules import console


def _print_help():
    print(__doc__)


def _cmd_record(argv):
    from modules.ui_cli import main as record_main
    record_main(argv)


def _cmd_gui(argv):
    from modules.ui_gui import main as gui_main
    gui_main(argv)


def _cmd_analyze(argv, quick=False):
    if quick:
        from modules.analyze import main as analyze_main
    else:
        from modules.extended import main as extended_main
        analyze_main = extended_main
    analyze_main(argv)


def _cmd_list(_argv):
    from modules.core import discover_device
    d = discover_device()
    if d:
        print(console.c_green("Найдено устройство:"), console.c_cyan(d))
    else:
        print(console.c_yellow("Устройство MERA не найдено"))


def _single_shot_file(argv):
    """Режим menu.py: python main.py -f FILE [--dir DIR] [--quick] + прочие опции анализа."""
    ap = argparse.ArgumentParser(description="Анализ одного файла стабилограммы")
    ap.add_argument("-f", "--file", default=None, help="Путь к файлу (CSV/XLSX)")
    ap.add_argument("--dir", default=None, help="Каталог для интерактивного выбора")
    ap.add_argument("--quick", action="store_true", help="Быстрый анализ вместо расширенного")
    known, rest = ap.parse_known_args(argv)
    analyzer_argv = list(rest)
    if known.file:
        analyzer_argv = [known.file] + analyzer_argv
    if known.dir and "--dir" not in analyzer_argv:
        analyzer_argv += ["--dir", known.dir]
    _cmd_analyze(analyzer_argv, quick=known.quick)


def _menu():
    default_records_dir()
    while True:
        print(console.c_bold("\n=== adStabilo ==="))
        print(f"  {console.c_cyan('1.')} Запись {console.c_dim('(консоль/matplotlib)')}")
        print(f"  {console.c_cyan('2.')} Запись {console.c_dim('(GUI Dear PyGui)')}")
        print(f"  {console.c_cyan('3.')} Анализ {console.c_green('расширенный')} {console.c_dim('(артефакты, Ромберг, нормы)')}")
        print(f"  {console.c_cyan('4.')} Анализ {console.c_green('быстрый')}")
        print(f"  {console.c_cyan('5.')} Список устройств")
        print(f"  {console.c_dim('0 / Q. Выход')}")
        try:
            choice = input("Выбор: ").strip().lower()
        except EOFError:
            break
        if not choice or choice in ("0", "q", "й"):
            break
        try:
            if choice == "1":
                _cmd_record([])
            elif choice == "2":
                _cmd_gui([])
            elif choice == "3":
                _cmd_analyze([])
            elif choice == "4":
                _cmd_analyze([], quick=True)
            elif choice == "5":
                _cmd_list([])
            else:
                print(console.c_red("Введите число 0–5."))
        except SystemExit:
            # --help внутри подмодуля или sys.exit: вернуться в меню
            pass
        except KeyboardInterrupt:
            print(console.c_yellow("\nОтменено, возврат в меню."))
        except Exception as e:
            print(console.c_red(f"Ошибка: {e}"))


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        _menu()
        return
    if argv[0] in ("-h", "--help", "help"):
        _print_help()
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "record":
        _cmd_record(rest)
    elif cmd == "gui":
        _cmd_gui(rest)
    elif cmd == "analyze":
        _cmd_analyze(rest)
    elif cmd == "quick":
        _cmd_analyze(rest, quick=True)
    elif cmd == "list":
        _cmd_list(rest)
    elif cmd in ("-f", "--file") or any(a in ("-f", "--file") for a in argv):
        _single_shot_file(argv)
    else:
        print(console.c_red(f"Неизвестная команда: {cmd}\n"))
        _print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
