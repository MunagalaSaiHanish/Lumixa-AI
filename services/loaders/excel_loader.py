import pandas as pd
from typing import Optional
from services.models.document import Document
from services.data_engine.dataset_registry import DatasetRegistry
from services.data_engine.data_cleaner import clean_dataframe


def load_excel(uploaded_file, registry: Optional[DatasetRegistry] = None) -> Optional[Document]:
    """
    Load an Excel file.

    Phase 2 behavior (preserved):
        Returns a Document with the sheet content as text for RAG ingestion.

    Phase 4 addition:
        If a DatasetRegistry is provided, also registers each sheet as a
        real Pandas DataFrame for structured data analysis.
        The DataFrames are stored in the registry under:
            registry[file_name][sheet_name] = DataFrame

    Args:
        uploaded_file: Streamlit UploadedFile object
        registry:      Optional DatasetRegistry to store DataFrames in
    """
    try:
        xls = pd.ExcelFile(uploaded_file)
    except Exception as e:
        print(f"[ExcelLoader] Failed to open file: {e}")
        return None

    source_name = uploaded_file.name
    full_text = ""

    for sheet_name in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name)
        except Exception as e:
            print(f"[ExcelLoader] Could not read sheet '{sheet_name}': {e}")
            continue

        if df.empty:
            continue

        # --- Phase 4: Register the real DataFrame for analysis ---
        if registry is not None:
            cleaned_df, clean_log = clean_dataframe(df)
            for entry in clean_log:
                print(f"[ExcelLoader] {entry}")
            registry.register(source_name, sheet_name, cleaned_df)

        # --- Phase 2: Convert to text for RAG ---
        sheet_text = f"[Sheet: {sheet_name}]\n"
        columns = [str(col).strip() for col in df.columns]
        rows_text = []

        for idx, row in df.iterrows():
            row_items = []
            for col, val in zip(columns, row):
                if pd.isna(val) or str(val).strip() == "":
                    continue
                row_items.append(f"{col}: {str(val).strip()}")
            if row_items:
                rows_text.append(", ".join(row_items))

        # Only include first 500 rows in RAG text to avoid token explosion
        if len(rows_text) > 500:
            shown = rows_text[:500]
            sheet_text += "\n".join(shown)
            sheet_text += f"\n[... {len(rows_text) - 500} more rows — use Data Analyst for full analysis]\n"
        else:
            sheet_text += "\n".join(rows_text) + "\n"

        full_text += sheet_text + "\n"

    if not full_text.strip():
        return None

    return Document.create(
        source="xlsx",
        title=source_name,
        content=full_text.strip(),
        metadata={"file": source_name, "source": "xlsx"}
    )
