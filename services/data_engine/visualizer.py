from typing import Any, Dict, List, Optional


def recommend_chart(
    analysis_type: str,
    result_data: List[Dict[str, Any]],
    x_col: Optional[str] = None,
    y_col: Optional[str] = None,
    title: str = "",
) -> Dict[str, Any]:
    """
    Given an analysis result, recommend the most appropriate chart type
    and return a structured dict that a React/Plotly component can render.

    Chart type selection rules:
    - ranking / group comparison → bar chart
    - time series / trend        → line chart
    - correlation                → scatter plot
    - distribution / anomaly     → histogram
    - part-to-whole (≤6 items)   → pie chart (sparingly)
    - target vs actual           → grouped bar chart
    - period comparison          → grouped bar chart
    """
    if not result_data:
        return {}

    # Determine chart type from analysis type
    chart_map = {
        "ranking": "bar",
        "group_sum": "bar",
        "group_mean": "bar",
        "group_count": "bar",
        "top_n": "bar",
        "bottom_n": "bar",
        "time_series": "line",
        "trend": "line",
        "date_group": "line",
        "period_compare": "bar",
        "correlation": "scatter",
        "distribution": "histogram",
        "anomaly": "scatter",
        "target_vs_actual": "bar",
        "scalar": None,  # No chart for single number results
        "auto_insights": "bar",
    }

    chart_type = chart_map.get(analysis_type, "bar")
    if chart_type is None:
        return {}

    # For pie charts: only use when ≤6 categories and analysis type is ranking/group
    if analysis_type in ("ranking", "group_sum") and len(result_data) <= 6 and len(result_data) >= 2:
        # Keep as bar — pie is rarely the best choice for business data
        pass

    # Infer x and y columns from the first record if not provided
    if result_data:
        keys = list(result_data[0].keys())
        if x_col is None and len(keys) >= 1:
            x_col = keys[0]
        if y_col is None and len(keys) >= 2:
            y_col = keys[1]

    return {
        "chart_type": chart_type,
        "title": title,
        "x_axis": x_col,
        "y_axis": y_col,
        "data": result_data,
    }
