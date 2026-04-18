from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_directories(project_root: Path) -> None:
    for relative in [
        "data/processed",
        "outputs/figures",
        "outputs/tables",
        "reports",
        ".cache/matplotlib",
    ]:
        (project_root / relative).mkdir(parents=True, exist_ok=True)


def load_reviews(config: dict) -> pd.DataFrame:
    source = Path(config["data"]["source_excel"])
    if not source.exists():
        raise FileNotFoundError(f"Excel source not found: {source}")
    return pd.read_excel(source, sheet_name=config["data"]["sheet_name"])


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_workbook(sheets: dict[str, pd.DataFrame], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            safe_name = sheet_name[:31]
            export_df = df.copy()
            for column in export_df.columns:
                if pd.api.types.is_datetime64tz_dtype(export_df[column]):
                    export_df[column] = export_df[column].dt.tz_convert(None)
            export_df.to_excel(writer, sheet_name=safe_name, index=False)
