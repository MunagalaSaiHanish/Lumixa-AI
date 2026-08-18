"""
Data Executor — Trusted Pandas Operations Only.

IMPORTANT SAFETY RULE:
    This module NEVER uses exec() or eval() with LLM-generated code.
    The LLM provides a plan (what to compute).
    This module performs the computation using trusted Python/Pandas code.
    The LLM never touches the actual data operation.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

# Try to import scipy for z-score. Fall back to manual z-score if not available.
try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[Executor] scipy not available. Z-score will use manual calculation.")


def _apply_filter(df: pd.DataFrame, filter_col: Optional[str], filter_value: Optional[Any]) -> Tuple[pd.DataFrame, List[str]]:
    """Apply an optional filter to the DataFrame. Returns filtered df and warnings."""
    warnings = []
    if not filter_col or filter_value is None:
        return df, warnings
    if filter_col not in df.columns:
        warnings.append(f"Filter column '{filter_col}' not found — filter skipped.")
        return df, warnings
    filtered = df[df[filter_col].astype(str).str.contains(str(filter_value), case=False, na=False)]
    if filtered.empty:
        warnings.append(f"No rows matched filter: {filter_col} = '{filter_value}'. Showing all data.")
        return df, warnings
    warnings.append(f"Filtered to rows where {filter_col} = '{filter_value}' ({len(filtered):,} rows).")
    return filtered, warnings


def _parse_date_column(df: pd.DataFrame, date_col: str) -> Tuple[pd.DataFrame, List[str]]:
    """Ensure a date column is datetime dtype. Returns updated df and warnings."""
    warnings = []
    if date_col not in df.columns:
        return df, [f"Date column '{date_col}' not found."]
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        try:
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce", infer_datetime_format=True)
            null_dates = df[date_col].isna().sum()
            if null_dates > 0:
                warnings.append(f"{null_dates} rows had unparseable dates in '{date_col}' and were excluded.")
            df = df.dropna(subset=[date_col])
        except Exception as e:
            warnings.append(f"Could not parse '{date_col}' as dates: {e}")
    return df, warnings


def execute_plan(df: pd.DataFrame, plan: Dict[str, Any]) -> Tuple[List[Dict], Optional[Any], str, List[str]]:
    """
    Execute an analysis plan on a DataFrame.

    Returns:
        result_data:   List of dicts (rows) for table/chart display
        scalar_result: Single number result (or None)
        analysis_type: The inferred analysis type string
        warnings:      List of warning messages
    """
    operation = plan.get("operation", "")
    target_col = plan.get("target_column")
    group_col = plan.get("group_by_column")
    secondary_col = plan.get("secondary_column")
    filter_col = plan.get("filter_column")
    filter_value = plan.get("filter_value")
    date_col = plan.get("date_column")
    date_period = plan.get("date_period", "month")
    period_a = plan.get("period_a")
    period_b = plan.get("period_b")
    sort_order = plan.get("sort_order", "desc")
    limit = plan.get("limit")
    analysis_type = plan.get("analysis_type", "scalar")

    ascending = (sort_order == "asc")
    warnings = []

    # Apply filter if specified
    df, filter_warnings = _apply_filter(df, filter_col, filter_value)
    warnings.extend(filter_warnings)

    if df.empty:
        return [], None, analysis_type, ["Dataset is empty after filtering."]

    # -----------------------------------------------------------------------
    # SCALAR OPERATIONS
    # -----------------------------------------------------------------------

    if operation == "SUM":
        total = df[target_col].sum()
        return [], round(float(total), 4), "scalar", warnings

    if operation == "COUNT":
        count_col = target_col or df.columns[0]
        count = int(df[count_col].count())
        return [], count, "scalar", warnings

    if operation == "MEAN":
        mean_val = df[target_col].mean()
        return [], round(float(mean_val), 4), "scalar", warnings

    if operation == "MEDIAN":
        median_val = df[target_col].median()
        return [], round(float(median_val), 4), "scalar", warnings

    if operation == "MIN":
        min_val = df[target_col].min()
        return [], round(float(min_val), 4), "scalar", warnings

    if operation == "MAX":
        max_val = df[target_col].max()
        return [], round(float(max_val), 4), "scalar", warnings

    if operation == "PERCENTAGE":
        # What percentage of total does each group represent
        if group_col and target_col:
            grouped = df.groupby(group_col)[target_col].sum().reset_index()
            total = grouped[target_col].sum()
            if total == 0:
                return [], None, "ranking", ["Total is zero — percentage cannot be computed."]
            grouped["percentage"] = (grouped[target_col] / total * 100).round(2)
            grouped = grouped.sort_values(target_col, ascending=ascending)
            result = grouped.to_dict(orient="records")
            return result, None, "ranking", warnings
        total = df[target_col].sum()
        return [], round(float(total), 4), "scalar", warnings

    # -----------------------------------------------------------------------
    # GROUP-BY OPERATIONS
    # -----------------------------------------------------------------------

    if operation in ("GROUP_BY_SUM", "TOP_N", "BOTTOM_N"):
        grouped = df.groupby(group_col)[target_col].sum().reset_index()
        grouped = grouped.sort_values(target_col, ascending=(operation == "BOTTOM_N"))
        if limit:
            grouped = grouped.head(int(limit))
        result = grouped.to_dict(orient="records")
        atype = "top_n" if operation == "TOP_N" else ("bottom_n" if operation == "BOTTOM_N" else "group_sum")
        return result, None, atype, warnings

    if operation == "GROUP_BY_MEAN":
        grouped = df.groupby(group_col)[target_col].mean().round(4).reset_index()
        grouped = grouped.sort_values(target_col, ascending=ascending)
        if limit:
            grouped = grouped.head(int(limit))
        return grouped.to_dict(orient="records"), None, "group_mean", warnings

    if operation == "GROUP_BY_COUNT":
        count_col = target_col or df.columns[0]
        grouped = df.groupby(group_col)[count_col].count().reset_index()
        grouped.columns = [group_col, "count"]
        grouped = grouped.sort_values("count", ascending=ascending)
        if limit:
            grouped = grouped.head(int(limit))
        return grouped.to_dict(orient="records"), None, "group_count", warnings

    # -----------------------------------------------------------------------
    # TARGET VS ACTUAL
    # -----------------------------------------------------------------------

    if operation == "TARGET_VS_ACTUAL":
        # actual column = target_col, quota/target column = secondary_col
        grouped_actual = df.groupby(group_col)[target_col].sum()
        grouped_target = df.groupby(group_col)[secondary_col].sum()
        combined = pd.DataFrame({
            group_col: grouped_actual.index,
            target_col: grouped_actual.values,
            secondary_col: grouped_target.values,
        })
        # Safe division — avoid division by zero
        combined["achievement_pct"] = combined.apply(
            lambda row: round((row[target_col] / row[secondary_col]) * 100, 2)
            if row[secondary_col] != 0 else None,
            axis=1
        )
        combined["met_target"] = combined["achievement_pct"].apply(
            lambda x: "Yes" if x is not None and x >= 100 else "No"
        )
        combined = combined.sort_values("achievement_pct", ascending=False)
        return combined.to_dict(orient="records"), None, "target_vs_actual", warnings

    # -----------------------------------------------------------------------
    # DATE-BASED OPERATIONS
    # -----------------------------------------------------------------------

    if operation in ("DATE_GROUP", "TREND"):
        df, date_warnings = _parse_date_column(df, date_col)
        warnings.extend(date_warnings)
        if df.empty:
            return [], None, "time_series", warnings

        # Map period to pandas frequency
        freq_map = {
            "day": "D",
            "week": "W",
            "month": "ME",
            "quarter": "QE",
            "year": "YE",
        }
        freq = freq_map.get(date_period or "month", "ME")

        df_indexed = df.set_index(date_col)
        grouped = df_indexed[target_col].resample(freq).sum().reset_index()
        grouped.columns = [date_col, target_col]
        grouped[date_col] = grouped[date_col].dt.strftime(
            "%Y-%m" if date_period in ("month", None) else
            "%Y-Q%q" if date_period == "quarter" else
            "%Y"
        )
        result = grouped.to_dict(orient="records")
        return result, None, "time_series", warnings

    if operation == "PERIOD_COMPARE":
        df, date_warnings = _parse_date_column(df, date_col)
        warnings.extend(date_warnings)
        if df.empty:
            return [], None, "period_compare", warnings

        df["_year"] = df[date_col].dt.year.astype(str)
        results = []
        for period in [str(period_a), str(period_b)]:
            subset = df[df["_year"] == period]
            if subset.empty:
                warnings.append(f"No data found for period '{period}'.")
                continue
            total = round(float(subset[target_col].sum()), 4)
            results.append({"period": period, target_col: total})

        if len(results) == 2:
            val_a = results[0][target_col]
            val_b = results[1][target_col]
            if val_a != 0:
                pct_change = round(((val_b - val_a) / abs(val_a)) * 100, 2)
                results.append({"period": "% Change", target_col: pct_change})

        return results, None, "period_compare", warnings

    # -----------------------------------------------------------------------
    # ANOMALY DETECTION
    # -----------------------------------------------------------------------

    if operation == "ANOMALY_ZSCORE":
        col_data = df[target_col].dropna()
        if SCIPY_AVAILABLE:
            z_scores = scipy_stats.zscore(col_data)
        else:
            mean = col_data.mean()
            std = col_data.std()
            z_scores = (col_data - mean) / std if std != 0 else col_data * 0

        threshold = 2.5
        anomaly_mask = (abs(z_scores) > threshold)
        # anomaly_mask may be a numpy array (from scipy) or pandas Series (from manual calc).
        # Normalize to a plain numpy boolean array for safe indexing.
        mask_array = np.asarray(anomaly_mask)
        anomalies = df.loc[col_data.index[mask_array]].copy()
        z_values = np.round(z_scores[mask_array], 3)
        anomalies["z_score"] = z_values if not hasattr(z_values, "values") else z_values.values

        if group_col and group_col in anomalies.columns:
            result_cols = [group_col, target_col, "z_score"]
        elif date_col and date_col in anomalies.columns:
            result_cols = [date_col, target_col, "z_score"]
        else:
            result_cols = [target_col, "z_score"]

        result = anomalies[result_cols].to_dict(orient="records")
        if not result:
            warnings.append(f"No statistical outliers detected (Z-score threshold: {threshold}).")
        return result, None, "anomaly", [
            f"Method: Z-score with threshold ±{threshold}",
            f"Detected {len(result)} outlier(s).",
        ] + warnings

    if operation == "ANOMALY_IQR":
        col_data = df[target_col].dropna()
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        anomalies = df[(df[target_col] < lower) | (df[target_col] > upper)].copy()
        anomalies["iqr_flag"] = anomalies[target_col].apply(
            lambda x: "below" if x < lower else "above"
        )
        if group_col and group_col in anomalies.columns:
            result_cols = [group_col, target_col, "iqr_flag"]
        elif date_col and date_col in anomalies.columns:
            result_cols = [date_col, target_col, "iqr_flag"]
        else:
            result_cols = [target_col, "iqr_flag"]
        result = anomalies[result_cols].to_dict(orient="records")
        if not result:
            warnings.append(f"No outliers detected (IQR method, bounds: {lower:.2f} – {upper:.2f}).")
        return result, None, "anomaly", [
            f"Method: IQR (Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f})",
            f"Bounds: [{lower:.2f}, {upper:.2f}]",
            f"Detected {len(result)} outlier(s).",
        ] + warnings

    # -----------------------------------------------------------------------
    # CORRELATION
    # -----------------------------------------------------------------------

    if operation == "CORRELATION":
        paired = df[[target_col, secondary_col]].dropna()
        if len(paired) < 3:
            return [], None, "correlation", ["Not enough data points for correlation (need at least 3)."]
        corr_value = round(float(paired[target_col].corr(paired[secondary_col])), 4)
        # Interpretation
        abs_corr = abs(corr_value)
        if abs_corr >= 0.7:
            strength = "strong"
        elif abs_corr >= 0.4:
            strength = "moderate"
        else:
            strength = "weak"
        direction = "positive" if corr_value >= 0 else "negative"
        interpretation = f"{strength} {direction} correlation"

        result_data = [
            {target_col: round(float(r[target_col]), 4), secondary_col: round(float(r[secondary_col]), 4)}
            for _, r in paired.head(200).iterrows()
        ]
        return result_data, corr_value, "correlation", [
            f"Pearson r = {corr_value} ({interpretation})",
            f"Sample size: {len(paired):,} rows",
        ]

    # -----------------------------------------------------------------------
    # AUTO INSIGHTS
    # -----------------------------------------------------------------------

    if operation == "AUTO_INSIGHTS":
        results = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        for num_col in numeric_cols[:3]:  # Top 3 numeric columns
            total = round(float(df[num_col].sum()), 2)
            mean_val = round(float(df[num_col].mean()), 2)
            results.append({
                "metric": f"Total {num_col}",
                "value": total,
                "insight": f"Mean per row: {mean_val}"
            })
            # Best category for this numeric col
            if cat_cols:
                best_cat_col = cat_cols[0]
                grouped = df.groupby(best_cat_col)[num_col].sum()
                best = grouped.idxmax()
                worst = grouped.idxmin()
                results.append({
                    "metric": f"Best {best_cat_col} by {num_col}",
                    "value": round(float(grouped[best]), 2),
                    "insight": best
                })
                results.append({
                    "metric": f"Lowest {best_cat_col} by {num_col}",
                    "value": round(float(grouped[worst]), 2),
                    "insight": worst
                })

        return results, None, "auto_insights", warnings

    # -----------------------------------------------------------------------
    # FILTER (return filtered rows)
    # -----------------------------------------------------------------------

    if operation == "FILTER":
        if limit:
            df = df.head(int(limit))
        cols = [c for c in [group_col, target_col, secondary_col, date_col] if c and c in df.columns]
        if not cols:
            cols = list(df.columns[:5])
        return df[cols].to_dict(orient="records"), None, "filter", warnings

    # Unknown operation — should not reach here after validation
    return [], None, analysis_type, [f"Operation '{operation}' is not implemented."]
