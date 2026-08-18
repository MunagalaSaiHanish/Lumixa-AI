import pandas as pd
from typing import Dict, List, Any


def inspect_schema(df: pd.DataFrame, source_name: str = "", sheet_name: str = "") -> Dict[str, Any]:
    """
    Inspect a DataFrame and return a detailed schema profile.

    This schema is used by:
    1. The Analysis Planner (to know what columns and types exist)
    2. The UI (to show the user a dataset overview)
    3. The Plan Validator (to validate column names before execution)

    Returns a dict with all the information the analysis engine needs.
    """
    schema = {
        "source_name": source_name,
        "sheet_name": sheet_name,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": [],         # list of column detail dicts
        "numeric_columns": [], # column names that are numeric
        "datetime_columns": [], # column names that look like dates
        "categorical_columns": [], # column names that are text/category
        "has_missing_values": False,
    }

    total_nulls = 0

    for col in df.columns:
        series = df[col]
        dtype_str = str(series.dtype)
        null_count = int(series.isna().sum())
        total_nulls += null_count
        unique_count = int(series.nunique(dropna=True))

        col_info = {
            "name": col,
            "dtype": dtype_str,
            "null_count": null_count,
            "unique_count": unique_count,
            "sample_values": [],
        }

        # Get a few sample values for context
        sample = series.dropna().head(3).tolist()
        col_info["sample_values"] = [str(v) for v in sample]

        # Classify column type
        if pd.api.types.is_numeric_dtype(series):
            col_info["column_type"] = "numeric"
            schema["numeric_columns"].append(col)
        elif pd.api.types.is_datetime64_any_dtype(series):
            col_info["column_type"] = "datetime"
            schema["datetime_columns"].append(col)
        else:
            # Try to detect date-like string columns by name and content
            col_lower = col.lower()
            is_date_name = any(kw in col_lower for kw in ["date", "month", "year", "quarter", "period", "time", "week"])
            if is_date_name and unique_count < len(df) * 0.95:
                # Try to parse a sample
                try:
                    pd.to_datetime(series.dropna().head(10))
                    col_info["column_type"] = "datetime_string"
                    schema["datetime_columns"].append(col)
                except Exception:
                    col_info["column_type"] = "categorical"
                    schema["categorical_columns"].append(col)
            else:
                col_info["column_type"] = "categorical"
                schema["categorical_columns"].append(col)

        schema["columns"].append(col_info)

    schema["has_missing_values"] = total_nulls > 0
    schema["total_null_count"] = total_nulls

    return schema


def schema_to_text(schema: Dict[str, Any]) -> str:
    """
    Convert a schema dict into a concise text description.
    This is what gets sent to the LLM Analysis Planner.
    We keep it short to save tokens.
    """
    lines = [
        f"Dataset: {schema['source_name']} / Sheet: {schema['sheet_name']}",
        f"Rows: {schema['row_count']:,}  |  Columns: {schema['column_count']}",
        "",
        "Columns:",
    ]
    for col in schema["columns"]:
        sample = ", ".join(col["sample_values"][:2])
        lines.append(
            f"  - {col['name']} [{col['column_type']}]  (sample: {sample})"
        )
    lines.append("")
    lines.append(f"Numeric columns : {', '.join(schema['numeric_columns']) or 'None'}")
    lines.append(f"Date columns    : {', '.join(schema['datetime_columns']) or 'None'}")
    lines.append(f"Category columns: {', '.join(schema['categorical_columns']) or 'None'}")
    if schema["has_missing_values"]:
        lines.append(f"Missing values  : {schema['total_null_count']:,} cells")
    return "\n".join(lines)
