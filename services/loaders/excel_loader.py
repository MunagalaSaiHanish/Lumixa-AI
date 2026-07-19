import pandas as pd
from services.models.document import Document

def load_excel(uploaded_file) -> Document:
    try:
        xls = pd.ExcelFile(uploaded_file)
    except Exception:
        return None

    full_text = ""
    for sheet_name in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name)
        except Exception:
            continue

        if df.empty:
            continue

        sheet_text = f"[Sheet: {sheet_name}]\n"
        rows_text = []
        columns = [str(col).strip() for col in df.columns]

        for idx, row in df.iterrows():
            row_items = []
            for col, val in zip(columns, row):
                if pd.isna(val) or str(val).strip() == "":
                    continue
                row_items.append(f"{col}: {str(val).strip()}")
            if row_items:
                rows_text.append(f"Row {idx + 1}: " + ", ".join(row_items))

        if rows_text:
            sheet_text += "\n".join(rows_text) + "\n"
            full_text += sheet_text + "\n"

    if not full_text.strip():
        return None

    return Document.create(
        source="xlsx",
        title=uploaded_file.name,
        content=full_text.strip(),
        metadata={"file": uploaded_file.name}
    )
