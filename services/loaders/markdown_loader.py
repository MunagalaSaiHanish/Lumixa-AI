import re
from services.models.document import Document

def load_markdown(uploaded_file) -> Document:
    encodings = ['utf-8', 'latin-1', 'cp1252']
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

    lines = text.split("\n")
    processed_lines = []
    for line in lines:
        stripped = line.strip()
        match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if match:
            heading_title = match.group(2).strip()
            processed_lines.append(f"\n[Section: {heading_title}]\n")
        else:
            processed_lines.append(line)

    return Document.create(
        source="md",
        title=uploaded_file.name,
        content="\n".join(processed_lines).strip(),
        metadata={"file": uploaded_file.name}
    )
