from typing import Any, Dict, List
from services.data_engine.analysis_planner import ALLOWED_OPERATIONS


def validate_plan(
    plan: Dict[str, Any],
    schema: Dict[str, Any],
) -> List[str]:
    """
    Validate an analysis plan against the real dataset schema.

    This runs BEFORE any data operation executes.
    If validation fails, we return errors and do NOT execute.

    Returns a list of error strings.
    An empty list means the plan is valid and safe to execute.
    """
    errors = []

    if not plan:
        errors.append("Analysis plan is empty.")
        return errors

    # --- Check operation is allowed ---
    operation = plan.get("operation", "")
    if operation not in ALLOWED_OPERATIONS:
        errors.append(
            f"Operation '{operation}' is not supported. "
            f"Allowed: {', '.join(ALLOWED_OPERATIONS)}"
        )

    # Collect all real column names from the schema
    real_columns = {col["name"] for col in schema.get("columns", [])}

    # --- Check target_column ---
    target_col = plan.get("target_column")
    if target_col:
        if target_col not in real_columns:
            errors.append(f"Target column '{target_col}' does not exist in the dataset.")
        else:
            # Check it's actually numeric
            numeric_cols = set(schema.get("numeric_columns", []))
            if target_col not in numeric_cols:
                errors.append(
                    f"Target column '{target_col}' is not numeric. "
                    f"Numeric columns: {', '.join(numeric_cols) or 'none'}"
                )

    # --- Check group_by_column ---
    group_col = plan.get("group_by_column")
    if group_col and group_col not in real_columns:
        errors.append(f"Group-by column '{group_col}' does not exist in the dataset.")

    # --- Check secondary_column ---
    secondary_col = plan.get("secondary_column")
    if secondary_col and secondary_col not in real_columns:
        errors.append(f"Secondary column '{secondary_col}' does not exist in the dataset.")

    # --- Check filter_column ---
    filter_col = plan.get("filter_column")
    if filter_col and filter_col not in real_columns:
        errors.append(f"Filter column '{filter_col}' does not exist in the dataset.")

    # --- Check date_column ---
    date_col = plan.get("date_column")
    if date_col and date_col not in real_columns:
        errors.append(f"Date column '{date_col}' does not exist in the dataset.")

    # --- Check date_period ---
    date_period = plan.get("date_period")
    allowed_periods = {None, "day", "week", "month", "quarter", "year"}
    if date_period not in allowed_periods:
        errors.append(f"date_period '{date_period}' is invalid. Use: day, week, month, quarter, year.")

    # --- Operation-specific requirements ---
    if operation in ("GROUP_BY_SUM", "GROUP_BY_MEAN", "GROUP_BY_COUNT"):
        if not target_col and operation != "GROUP_BY_COUNT":
            errors.append(f"Operation {operation} requires a target_column.")
        if not group_col:
            errors.append(f"Operation {operation} requires a group_by_column.")

    if operation in ("TOP_N", "BOTTOM_N"):
        if not target_col:
            errors.append(f"Operation {operation} requires a target_column.")
        if not group_col:
            errors.append(f"Operation {operation} requires a group_by_column.")

    if operation == "CORRELATION":
        if not target_col:
            errors.append("CORRELATION requires a target_column.")
        if not secondary_col:
            errors.append("CORRELATION requires a secondary_column.")

    if operation == "TARGET_VS_ACTUAL":
        if not target_col:
            errors.append("TARGET_VS_ACTUAL requires a target_column (actual sales column).")
        if not secondary_col:
            errors.append("TARGET_VS_ACTUAL requires a secondary_column (target/quota column).")
        if not group_col:
            errors.append("TARGET_VS_ACTUAL requires a group_by_column (e.g. Employee, Region).")

    if operation in ("DATE_GROUP", "TREND", "PERIOD_COMPARE"):
        if not date_col:
            errors.append(f"Operation {operation} requires a date_column.")
        if not target_col:
            errors.append(f"Operation {operation} requires a target_column.")

    if operation == "PERIOD_COMPARE":
        if not plan.get("period_a") or not plan.get("period_b"):
            errors.append("PERIOD_COMPARE requires period_a and period_b (e.g. '2024', '2025').")

    return errors
