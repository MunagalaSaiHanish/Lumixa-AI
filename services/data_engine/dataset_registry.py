import pandas as pd
from typing import Dict, List, Optional, Tuple


class DatasetRegistry:
    """
    Stores loaded DataFrames in memory for the lifetime of a session.

    When a user uploads an Excel or CSV file, the loaders call
    registry.register() to save the DataFrame here.

    Later, when the user asks an analytical question, the Data Engine
    calls registry.get_dataframe() to retrieve the real DataFrame
    and perform actual Pandas calculations on it.

    Structure:
        self._store = {
            "sales.xlsx": {
                "Sheet1": <DataFrame>,
                "Targets": <DataFrame>,
            },
            "employees.csv": {
                "default": <DataFrame>,
            }
        }
    """

    def __init__(self):
        # source_name → { sheet_name → DataFrame }
        self._store: Dict[str, Dict[str, pd.DataFrame]] = {}

    def register(self, source_name: str, sheet_name: str, df: pd.DataFrame):
        """
        Save a DataFrame for a given source file and sheet.

        Args:
            source_name: The file name (e.g. "sales.xlsx")
            sheet_name:  The sheet name (e.g. "Monthly Sales") or "default" for CSV
            df:          The Pandas DataFrame to store
        """
        if source_name not in self._store:
            self._store[source_name] = {}
        self._store[source_name][sheet_name] = df
        print(f"[Registry] Registered '{source_name}' / '{sheet_name}' — {len(df)} rows × {len(df.columns)} cols")

    def get_dataframe(self, source_name: str, sheet_name: str) -> Optional[pd.DataFrame]:
        """Retrieve a stored DataFrame. Returns None if not found."""
        return self._store.get(source_name, {}).get(sheet_name)

    def get_all_sheets(self, source_name: str) -> Dict[str, pd.DataFrame]:
        """Return all sheet DataFrames for a given source file."""
        return self._store.get(source_name, {})

    def list_datasets(self) -> List[Dict[str, any]]:
        """
        Return a summary list of all registered datasets.
        Useful for showing the user what is available.
        """
        summary = []
        for source_name, sheets in self._store.items():
            for sheet_name, df in sheets.items():
                summary.append({
                    "source_name": source_name,
                    "sheet_name": sheet_name,
                    "rows": len(df),
                    "columns": list(df.columns),
                    "column_count": len(df.columns),
                })
        return summary

    def has_datasets(self) -> bool:
        """Returns True if at least one dataset is registered."""
        return len(self._store) > 0

    def find_best_dataset(self, question: str) -> Optional[Tuple[str, str]]:
        """
        Simple heuristic: if only one dataset exists, return it.
        If multiple exist, try to match the question to a source name.
        Returns (source_name, sheet_name) or None if ambiguous/empty.
        """
        all_datasets = self.list_datasets()
        if not all_datasets:
            return None
        if len(all_datasets) == 1:
            d = all_datasets[0]
            return (d["source_name"], d["sheet_name"])
        # Try to match question keywords against source names
        question_lower = question.lower()
        for d in all_datasets:
            name_lower = d["source_name"].lower().replace("_", " ").replace("-", " ")
            # Remove extension
            name_lower = name_lower.rsplit(".", 1)[0]
            if any(word in question_lower for word in name_lower.split()):
                return (d["source_name"], d["sheet_name"])
        # If still ambiguous, return the first one and let the planner handle it
        d = all_datasets[0]
        return (d["source_name"], d["sheet_name"])

    def clear(self):
        """Remove all stored DataFrames. Called when workspace is cleared."""
        self._store.clear()
        print("[Registry] All datasets cleared.")
