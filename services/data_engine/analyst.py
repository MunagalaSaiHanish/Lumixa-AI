"""
analyst.py — Main Data Analyst Orchestrator (Phase 4)

Pipeline:
    User Question
        → Dataset Selection (from registry)
        → Schema Inspection
        → Analysis Planning (LLM produces JSON plan)
        → Plan Validation (no exec/eval — safety check)
        → Data Execution (trusted Pandas operations)
        → LLM Explanation (explains verified numbers)
        → AnalysisResult (returned to the UI)
"""

import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from config import LLM_MODEL, TEMPERATURE
from services.data_engine.dataset_registry import DatasetRegistry
from services.data_engine.schema_inspector import inspect_schema, schema_to_text
from services.data_engine.data_cleaner import clean_dataframe
from services.data_engine.analysis_planner import create_analysis_plan
from services.data_engine.plan_validator import validate_plan
from services.data_engine.data_executor import execute_plan
from services.data_engine.visualizer import recommend_chart
from services.data_engine.result_model import AnalysisResult

load_dotenv()

_client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    timeout=120
)

EXPLAINER_SYSTEM = (
    "You are Lumixa AI, a business intelligence assistant. "
    "You will receive computed data results. "
    "Your job is to explain these results clearly and professionally. "
    "Use numbers from the provided results — do NOT invent or modify any numbers. "
    "Be concise and insightful. Use bullet points where appropriate."
)


def _generate_explanation(question: str, result_summary: str) -> str:
    """
    Ask the LLM to explain the computed results in natural language.
    The LLM receives the verified result — it only explains, never computes.
    """
    prompt = f"""
The user asked: "{question}"

Here are the computed results from the dataset:

{result_summary}

Write a clear, professional explanation of these results.
- Use the actual numbers provided above.
- Do not invent or modify any numbers.
- Highlight the most important finding first.
- Keep it concise (3-6 sentences or bullet points).
"""
    try:
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": EXPLAINER_SYSTEM},
                {"role": "user", "content": prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=500
        )
        content = response.choices[0].message.content or ""
        # Strip think tokens if present (Qwen model)
        import re
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content
    except Exception as e:
        print(f"[Analyst] Explanation generation failed: {e}")
        return "Analysis complete. See the results above."


def _format_result_for_explanation(result_data: list, scalar_result, analysis_type: str, plan: dict) -> str:
    """
    Format the computed result as a concise text summary to give to the LLM.
    We never send the full DataFrame — only the computed aggregates.
    This protects privacy and controls token usage.
    """
    lines = []

    if scalar_result is not None:
        target = plan.get("target_column", "value")
        lines.append(f"Result: {target} = {scalar_result:,}")

    if result_data:
        # Show top 15 rows maximum — never the full dataset
        display_rows = result_data[:15]
        for row in display_rows:
            row_str = " | ".join(f"{k}: {v}" for k, v in row.items())
            lines.append(row_str)
        if len(result_data) > 15:
            lines.append(f"... ({len(result_data) - 15} more rows not shown)")

    return "\n".join(lines) if lines else "No results computed."


def analyze(
    question: str,
    registry: DatasetRegistry,
    dataset_name: Optional[str] = None,
    sheet_name: Optional[str] = None,
) -> AnalysisResult:
    """
    Main entry point for structured data analysis.

    Args:
        question:     The user's natural language question.
        registry:     The DatasetRegistry holding all loaded DataFrames.
        dataset_name: Optional — which dataset to use. Auto-selected if None.
        sheet_name:   Optional — which sheet to use. Auto-selected if None.

    Returns:
        An AnalysisResult containing data, explanation, and visualization spec.
    """

    # --- Step 1: Select dataset ---
    if not registry.has_datasets():
        return AnalysisResult(
            question=question,
            dataset_name="",
            sheet_name="",
            analysis_type="error",
            explanation="No structured dataset is loaded. Please upload an Excel or CSV file first.",
            warnings=["No datasets in registry."]
        )

    if dataset_name is None or sheet_name is None:
        match = registry.find_best_dataset(question)
        if match is None:
            return AnalysisResult(
                question=question,
                dataset_name="",
                sheet_name="",
                analysis_type="error",
                explanation="Could not identify which dataset to use. Please specify.",
                warnings=["Dataset selection ambiguous."]
            )
        dataset_name, sheet_name = match

    df = registry.get_dataframe(dataset_name, sheet_name)
    if df is None or df.empty:
        return AnalysisResult(
            question=question,
            dataset_name=dataset_name,
            sheet_name=sheet_name,
            analysis_type="error",
            explanation=f"Dataset '{dataset_name}' / '{sheet_name}' is empty or was not found.",
            warnings=["Empty dataset."]
        )

    print(f"[Analyst] Dataset: '{dataset_name}' / '{sheet_name}' — {len(df):,} rows")

    # --- Step 2: Clean data ---
    df, clean_log = clean_dataframe(df)
    for entry in clean_log:
        print(f"[Cleaner] {entry}")

    # --- Step 3: Inspect schema ---
    schema = inspect_schema(df, source_name=dataset_name, sheet_name=sheet_name)
    schema_text = schema_to_text(schema)
    print(f"[Analyst] Schema: {len(schema['columns'])} columns | "
          f"Numeric: {len(schema['numeric_columns'])} | Date: {len(schema['datetime_columns'])}")

    # --- Step 4: Plan the analysis ---
    plan = create_analysis_plan(
        question=question,
        schema_text=schema_text,
        dataset_name=dataset_name,
        sheet_name=sheet_name
    )
    if plan is None:
        return AnalysisResult(
            question=question,
            dataset_name=dataset_name,
            sheet_name=sheet_name,
            analysis_type="error",
            explanation="I could not understand the analytical intent of your question. Please rephrase it.",
            warnings=["Analysis planning failed — LLM did not return a valid plan."]
        )

    # --- Step 5: Validate the plan ---
    validation_errors = validate_plan(plan, schema)
    if validation_errors:
        error_list = "\n".join(f"- {e}" for e in validation_errors)
        return AnalysisResult(
            question=question,
            dataset_name=dataset_name,
            sheet_name=sheet_name,
            analysis_type="error",
            explanation=f"The analysis plan has issues:\n{error_list}",
            warnings=validation_errors
        )

    # --- Step 6: Execute ---
    try:
        result_data, scalar_result, analysis_type, exec_warnings = execute_plan(df, plan)
    except Exception as e:
        print(f"[Analyst] Execution error: {e}")
        return AnalysisResult(
            question=question,
            dataset_name=dataset_name,
            sheet_name=sheet_name,
            analysis_type="error",
            explanation=f"Data computation failed: {e}",
            warnings=[str(e)]
        )

    # --- Step 7: Generate visualization spec ---
    chart_title = plan.get("chart_title", question[:60])
    target_col = plan.get("target_column")
    group_col = plan.get("group_by_column")
    date_col = plan.get("date_column")
    x_col = group_col or date_col or (list(result_data[0].keys())[0] if result_data else None)

    visualization = recommend_chart(
        analysis_type=analysis_type,
        result_data=result_data,
        x_col=x_col,
        y_col=target_col,
        title=chart_title,
    )

    # --- Step 8: LLM Explanation ---
    result_summary = _format_result_for_explanation(result_data, scalar_result, analysis_type, plan)
    explanation = _generate_explanation(question, result_summary)

    # Combine execution warnings with clean log (if any cleaning happened)
    all_warnings = exec_warnings + (clean_log if clean_log else [])

    print(f"[Analyst] Done — {len(result_data)} result rows, scalar={scalar_result}")

    return AnalysisResult(
        question=question,
        dataset_name=dataset_name,
        sheet_name=sheet_name,
        analysis_type=analysis_type,
        result_data=result_data,
        scalar_result=scalar_result,
        explanation=explanation,
        visualization=visualization,
        warnings=all_warnings,
        metadata={
            "operation": plan.get("operation"),
            "target_column": target_col,
            "group_by_column": group_col,
            "rows_analyzed": len(df),
        }
    )
