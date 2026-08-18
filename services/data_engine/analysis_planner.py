import json
import os
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

from config import LLM_MODEL, TEMPERATURE

load_dotenv()

# Use the same OpenRouter client as the rest of LUMIXA
_client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    timeout=120
)


# All operations the data engine supports.
# The LLM picks one of these — nothing else is accepted.
ALLOWED_OPERATIONS = [
    "SUM",
    "COUNT",
    "MEAN",
    "MEDIAN",
    "MIN",
    "MAX",
    "GROUP_BY_SUM",
    "GROUP_BY_MEAN",
    "GROUP_BY_COUNT",
    "TOP_N",
    "BOTTOM_N",
    "FILTER",
    "PERCENTAGE",
    "TARGET_VS_ACTUAL",
    "DATE_GROUP",
    "PERIOD_COMPARE",
    "TREND",
    "ANOMALY_ZSCORE",
    "ANOMALY_IQR",
    "CORRELATION",
    "AUTO_INSIGHTS",
]

PLANNER_SYSTEM = (
    "You are a data analysis planner for a business intelligence system. "
    "Your job is to read a user question and a dataset schema, then return "
    "a single valid JSON object describing the analysis plan. "
    "Return ONLY the JSON object — no markdown, no explanation, no code blocks."
)

PLANNER_TEMPLATE = """
User question:
{question}

Dataset schema:
{schema_text}

Available operations:
{operations}

Return a JSON object with exactly these fields:

{{
  "operation": "<one of the allowed operations>",
  "target_column": "<numeric column to measure, or null>",
  "group_by_column": "<column to group by, or null>",
  "secondary_column": "<second numeric column for correlation or target_vs_actual, or null>",
  "filter_column": "<column to filter on, or null>",
  "filter_value": "<value to filter for, or null>",
  "date_column": "<date/time column, or null>",
  "date_period": "<'day', 'week', 'month', 'quarter', 'year', or null>",
  "period_a": "<first period for comparison, e.g. '2024', or null>",
  "period_b": "<second period for comparison, e.g. '2025', or null>",
  "sort_order": "<'desc' or 'asc', default 'desc'>",
  "limit": <integer top N / bottom N, or null>,
  "analysis_type": "<short label: ranking, group_sum, time_series, target_vs_actual, scalar, correlation, anomaly, period_compare, trend, auto_insights>",
  "chart_title": "<a short human-readable chart title>"
}}

Rules:
- Choose the operation that best answers the question.
- target_column must be a numeric column from the schema.
- group_by_column must be a categorical or date column from the schema.
- If the question is about totals or sums → SUM or GROUP_BY_SUM.
- If the question is about averages → MEAN or GROUP_BY_MEAN.
- If the question is about top/bottom performers → TOP_N or BOTTOM_N with GROUP_BY_SUM first if needed.
- If the question mentions "target" and "actual" → TARGET_VS_ACTUAL.
- If the question is about trends over time → DATE_GROUP or TREND.
- If the question compares two years → PERIOD_COMPARE.
- If the question asks for unusual or anomaly values → ANOMALY_ZSCORE.
- If the question asks for correlation → CORRELATION.
- If the question asks for an overview or insights → AUTO_INSIGHTS.
- Only use column names that exist exactly in the schema.
- Return null for any field that is not needed for this operation.
"""


def create_analysis_plan(
    question: str,
    schema_text: str,
    dataset_name: str = "",
    sheet_name: str = "",
) -> Optional[Dict[str, Any]]:
    """
    Ask the LLM to produce a structured analysis plan for the user's question.

    Returns a dict with the plan, or None if planning fails.
    The plan is validated separately in plan_validator.py before execution.
    """
    prompt = PLANNER_TEMPLATE.format(
        question=question,
        schema_text=schema_text,
        operations=", ".join(ALLOWED_OPERATIONS)
    )

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": prompt}
    ]

    try:
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.0,  # Zero temperature — we want deterministic structured output
            max_tokens=600
        )
        content = response.choices[0].message.content or ""
        content = content.strip()

        # Strip any markdown fences the model might add despite instructions
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()

        plan = json.loads(content)

        # Inject dataset info
        plan["dataset_name"] = dataset_name
        plan["sheet_name"] = sheet_name
        plan["original_question"] = question

        print(f"[Planner] Operation selected: {plan.get('operation')} | Type: {plan.get('analysis_type')}")
        return plan

    except json.JSONDecodeError as e:
        print(f"[Planner] Could not parse LLM response as JSON: {e}")
        print(f"[Planner] Raw response: {content[:300]}")
        return None
    except Exception as e:
        print(f"[Planner] Planning failed: {e}")
        return None
