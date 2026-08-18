import numpy as np
import pandas as pd
from typing import Any, Dict, List


def generate_insights(df: pd.DataFrame, source_name: str, sheet_name: str) -> List[Dict[str, Any]]:
    """
    Automatically compute useful insights from any structured dataset.

    This runs actual Pandas computations — the LLM does NOT guess these.
    The results are shown to the user immediately after they upload a dataset,
    before they ask any questions.

    Returns a list of insight dicts, each containing:
        - title: short label
        - value: the computed number or string
        - detail: extra context
    """
    insights = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

    # --- Basic stats for each numeric column ---
    for col in numeric_cols[:4]:  # Limit to first 4 to keep UI clean
        col_data = df[col].dropna()
        if col_data.empty:
            continue
        total = col_data.sum()
        mean_val = col_data.mean()
        insights.append({
            "title": f"Total {col}",
            "value": f"{total:,.2f}",
            "detail": f"Mean: {mean_val:,.2f}  |  Rows: {len(col_data):,}"
        })

    # --- Best and worst per category ---
    if numeric_cols and cat_cols:
        main_num = numeric_cols[0]
        main_cat = cat_cols[0]
        grouped = df.groupby(main_cat)[main_num].sum()
        if not grouped.empty:
            best_name = grouped.idxmax()
            best_val = grouped.max()
            worst_name = grouped.idxmin()
            worst_val = grouped.min()
            insights.append({
                "title": f"Best {main_cat} by {main_num}",
                "value": str(best_name),
                "detail": f"{best_val:,.2f}"
            })
            insights.append({
                "title": f"Lowest {main_cat} by {main_num}",
                "value": str(worst_name),
                "detail": f"{worst_val:,.2f}"
            })

    # --- Date range info ---
    if date_cols:
        date_col = date_cols[0]
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        insights.append({
            "title": "Date Range",
            "value": f"{min_date.strftime('%Y-%m-%d')} → {max_date.strftime('%Y-%m-%d')}",
            "detail": f"Column: {date_col}"
        })

    # --- Target achievement (look for columns named with "target" or "quota") ---
    target_col_candidates = [c for c in numeric_cols if "target" in c.lower() or "quota" in c.lower() or "goal" in c.lower()]
    actual_col_candidates = [c for c in numeric_cols if "actual" in c.lower() or "sales" in c.lower() or "revenue" in c.lower() or "achieved" in c.lower()]

    if target_col_candidates and actual_col_candidates:
        target_col = target_col_candidates[0]
        actual_col = actual_col_candidates[0]
        total_actual = df[actual_col].sum()
        total_target = df[target_col].sum()
        if total_target > 0:
            achievement = round((total_actual / total_target) * 100, 1)
            insights.append({
                "title": "Overall Target Achievement",
                "value": f"{achievement}%",
                "detail": f"Actual: {total_actual:,.2f}  |  Target: {total_target:,.2f}"
            })

    # --- Dataset size ---
    insights.append({
        "title": "Dataset Size",
        "value": f"{len(df):,} rows × {len(df.columns)} columns",
        "detail": f"Source: {source_name}  |  Sheet: {sheet_name}"
    })

    return insights
