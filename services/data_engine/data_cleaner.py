import pandas as pd
from typing import List, Tuple


def clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply conservative cleaning to a DataFrame.

    Rules:
    - We do NOT silently drop rows unless they are completely empty.
    - We do NOT remove duplicates automatically (business data may have
      identical rows for different time periods).
    - We DO strip whitespace from column headers.
    - We DO try to coerce object columns that look numeric.
    - We DO try to parse object columns that look like dates.
    - We log every transformation made.

    Returns:
        cleaned_df: The cleaned DataFrame
        log: List of human-readable messages about what was changed
    """
    log = []
    df = df.copy()

    # --- Step 1: Clean column headers ---
    original_columns = list(df.columns)
    df.columns = [str(col).strip() for col in df.columns]
    new_columns = list(df.columns)
    changed = [(o, n) for o, n in zip(original_columns, new_columns) if o != n]
    for old, new in changed:
        log.append(f"Column header cleaned: '{old}' → '{new}'")

    # --- Step 2: Remove fully empty rows ---
    before = len(df)
    df = df.dropna(how="all")
    removed = before - len(df)
    if removed > 0:
        log.append(f"Removed {removed} fully empty rows.")

    # --- Step 3: Try to coerce object/string columns that look numeric ---
    for col in df.columns:
        # In Pandas 2.x, string columns may show as 'object' or 'string' dtype.
        # is_object_dtype covers the classic 'object' case.
        # We also check for string dtype explicitly for Pandas 2.x compatibility.
        col_is_text = (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
        )
        # But skip if it's already numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            col_is_text = False

        if col_is_text:
            # Try converting — if more than 80% succeed, accept it
            converted = pd.to_numeric(df[col].astype(str).str.replace(",", "").str.strip(), errors="coerce")
            valid_count = converted.notna().sum()
            original_valid = df[col].notna().sum()
            if original_valid > 0 and valid_count / original_valid >= 0.8:
                df[col] = converted
                log.append(f"Column '{col}' converted from text to numeric.")

    # --- Step 4: Try to parse object/string columns that look like dates ---
    for col in df.columns:
        col_is_text2 = (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_string_dtype(df[col])
        )
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            col_is_text2 = False

        if col_is_text2:
            col_lower = col.lower()
            is_date_name = any(kw in col_lower for kw in ["date", "month", "year", "quarter", "period", "time"])
            if is_date_name:
                try:
                    parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                    valid_ratio = parsed.notna().sum() / max(df[col].notna().sum(), 1)
                    if valid_ratio >= 0.8:
                        df[col] = parsed
                        log.append(f"Column '{col}' parsed as datetime.")
                except Exception:
                    pass

    return df, log
