from services.models.document import Document

def load_txt(uploaded_file) -> Document:
    encodings = ['utf-8', 'latin-1', 'cp1252', 'utf-16']
    text = ""
    for enc in encodings:
        try:
            uploaded_file.seek(0)
            text = uploaded_file.read().decode(enc).strip()
            if text:
                break
        except Exception:
            continue

    if not text:
        return None

    return Document.create(
        source="txt",
        title=uploaded_file.name,
        content=text,
        metadata={"file": uploaded_file.name}
    )
