import pandas as pd
from typing import Optional, Tuple
import re
import unicodedata

def _strip_accents(s: str) -> str:
    s = str(s)
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Keep original columns, but provide a normalized lookup if needed elsewhere.
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _try_read_csv(uploaded_file) -> pd.DataFrame:
    # Try common encodings in Mexico datasets and auto-detect delimiter (, or ;)
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=enc, sep=None, engine="python")
        except Exception as e:
            last_err = e
            continue
    raise last_err

def load_data(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = _try_read_csv(uploaded_file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Formato no soportado. Usa CSV o Excel.")
    return normalize_columns(df)

def parse_numeric_series(s: pd.Series) -> pd.Series:
    # Tolerant numeric parsing for strings like "$1,234.56" or "1.234,56"
    x = s.astype(str).str.strip()

    # Remove currency and spaces
    x = x.str.replace(r"[\$\s]", "", regex=True)

    # If it looks like European decimal (comma) with dot thousands: 1.234,56 -> 1234.56
    euro_mask = x.str.contains(r"\d+\.\d{3},\d+")
    x.loc[euro_mask] = x.loc[euro_mask].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)

    # If it looks like comma decimal without dots: 1234,56 -> 1234.56
    comma_dec_mask = x.str.contains(r"\d+,\d+$") & ~x.str.contains(r"\.")
    x.loc[comma_dec_mask] = x.loc[comma_dec_mask].str.replace(",", ".", regex=False)

    # Remove remaining thousands separators (commas) like 1,234.56 -> 1234.56
    x = x.str.replace(",", "", regex=False)

    return pd.to_numeric(x, errors="coerce")

def detect_catalog_error_description(df: pd.DataFrame) -> Tuple[bool, Optional[str], Optional[str]]:
    # Detect a catalog like: ERROR, DESCRIPCIÓN (accents/case-insensitive)
    cols_norm = [_strip_accents(c).upper() for c in df.columns]
    try:
        error_col = df.columns[cols_norm.index("ERROR")]
    except ValueError:
        error_col = None

    desc_col = None
    for i, c in enumerate(cols_norm):
        if c in ("DESCRIPCION", "DESCRIPCION_ERROR", "DESCRIPCION DE ERROR", "DESCRIPCION_DEL_ERROR"):
            desc_col = df.columns[i]
            break

    is_catalog = (error_col is not None) and (desc_col is not None) and (df.shape[1] <= 5)
    return is_catalog, error_col, desc_col

def build_series_sum(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    group_col: Optional[str],
    group_value: Optional[str],
    freq: str = "D",
    fill_missing: str = "zeros",
) -> pd.DataFrame:
    data = df.copy()

    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col])

    data[target_col] = parse_numeric_series(data[target_col])
    data = data.dropna(subset=[target_col])

    if group_col:
        data[group_col] = data[group_col].astype(str)
        if group_value is not None:
            data = data[data[group_col] == str(group_value)]

    data = data.sort_values(date_col)

    ts = (
        data.set_index(date_col)[target_col]
        .resample(freq)
        .sum()
        .to_frame("y")
    )

    if fill_missing == "zeros":
        ts["y"] = ts["y"].fillna(0.0)
    elif fill_missing == "ffill":
        ts["y"] = ts["y"].ffill().bfill()
    elif fill_missing == "interpolate":
        ts["y"] = ts["y"].interpolate(method="time").bfill().ffill()
    else:
        raise ValueError("fill_missing debe ser: zeros | ffill | interpolate")

    ts["y"] = ts["y"].clip(lower=0.0)
    ts.index.name = "ds"
    return ts.reset_index().set_index("ds")

def build_series_count(
    df: pd.DataFrame,
    date_col: str,
    event_col: str,
    event_value: Optional[str],
    group_col: Optional[str],
    group_value: Optional[str],
    freq: str = "D",
    fill_missing: str = "zeros",
) -> pd.DataFrame:
    """Build a series counting events (e.g., error codes) over time."""
    data = df.copy()

    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col])

    if group_col:
        data[group_col] = data[group_col].astype(str)
        if group_value is not None:
            data = data[data[group_col] == str(group_value)]

    data[event_col] = data[event_col].astype(str)
    if event_value is not None:
        data = data[data[event_col] == str(event_value)]

    data = data.sort_values(date_col)

    ts = (
        data.set_index(date_col)[event_col]
        .resample(freq)
        .count()
        .to_frame("y")
    )

    if fill_missing == "zeros":
        ts["y"] = ts["y"].fillna(0.0)
    elif fill_missing == "ffill":
        ts["y"] = ts["y"].ffill().bfill()
    elif fill_missing == "interpolate":
        ts["y"] = ts["y"].interpolate(method="time").bfill().ffill()
    else:
        raise ValueError("fill_missing debe ser: zeros | ffill | interpolate")

    ts["y"] = ts["y"].clip(lower=0.0)
    ts.index.name = "ds"
    return ts.reset_index().set_index("ds")
