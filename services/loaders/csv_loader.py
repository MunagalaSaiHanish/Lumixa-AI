import pandas as pd
from services.models.document import Document

def load_csv(uploaded_file) -> Document:
    df = None
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
    except Exception:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='latin1')
        except Exception:
            return None

    if df is None or df.empty:
        return None

    columns = [str(col).strip() for col in df.columns]
    rows_text = []

    for idx, row in df.iterrows():
        row_items = []
        for col, val in zip(columns, row):
            if pd.isna(val) or str(val).strip() == "":
                continue
            row_items.append(f"{col}: {str(val).strip()}")
        if row_items:
            rows_text.append(f"Row {idx + 1}: " + ", ".join(row_items))

    if not rows_text:
        return None

    return Document.create(
        source="csv",
        title=uploaded_file.name,
        content="\n".join(rows_text),
        metadata={"file": uploaded_file.name}
    )
