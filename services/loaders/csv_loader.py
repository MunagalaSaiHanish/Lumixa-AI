import pandas as pd
from typing import Optional
from services.models.document import Document
from services.data_engine.dataset_registry import DatasetRegistry
from services.data_engine.data_cleaner import clean_dataframe


def load_csv(uploaded_file, registry: Optional[DatasetRegistry] = None) -> Optional[Document]:
    """
    Load a CSV file.

    Phase 2 behavior (preserved):
        Returns a Document with row content as text for RAG ingestion.

    Phase 4 addition:
        If a DatasetRegistry is provided, also registers the DataFrame
        for structured data analysis under:
            registry[file_name]["default"] = DataFrame

    Args:
        uploaded_file: Streamlit UploadedFile object
        registry:      Optional DatasetRegistry to store the DataFrame in
    """
    df = None
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
    except Exception:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="latin1")
        except Exception as e:
            print(f"[CSVLoader] Failed to read file: {e}")
            return None

    if df is None or df.empty:
        return None

    source_name = uploaded_file.name

    # --- Phase 4: Register the real DataFrame for analysis ---
    if registry is not None:
        cleaned_df, clean_log = clean_dataframe(df)
        for entry in clean_log:
            print(f"[CSVLoader] {entry}")
        registry.register(source_name, "default", cleaned_df)

    # --- Phase 2: Convert to text for RAG ---
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

    if not rows_text:
        return None

    # Limit RAG text to 500 rows to avoid token explosion
    if len(rows_text) > 500:
        rag_text = "\n".join(rows_text[:500])
        rag_text += f"\n[... {len(rows_text) - 500} more rows — use Data Analyst for full analysis]"
    else:
        rag_text = "\n".join(rows_text)

    return Document.create(
        source="csv",
        title=source_name,
        content=rag_text,
        metadata={"file": source_name, "source": "csv"}
    )
