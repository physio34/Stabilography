"""Общие константы adStabilo: корень проекта и каталог записей."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RECORDS_DIR = ROOT_DIR / "records"

SUPPORTED_STABILO_EXTS = (".csv", ".xlsx", ".xls")

DEFAULT_OUT_NAME = "stabilo.csv"


def default_records_dir() -> Path:
    """Возвращает каталог records/, создавая его при необходимости."""
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    return RECORDS_DIR


def resolve_out_path(out: str | Path | None) -> Path:
    """Превращает --out в путь: bare-имя кладётся в records/."""
    if not out:
        return default_records_dir() / DEFAULT_OUT_NAME
    p = Path(out)
    if not p.is_absolute() and len(p.parts) == 1:
        return default_records_dir() / p
    return p
