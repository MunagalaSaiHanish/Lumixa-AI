import docx
from docx.text.paragraph import Paragraph
from docx.table import Table
from services.models.document import Document

def iter_block_items(parent):
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    for child in parent.element.body:
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)

def load_docx(uploaded_file) -> Document:
    try:
        doc = docx.Document(uploaded_file)
    except Exception:
        return None

    full_text = ""
    current_section = "General"

    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            if block.style and block.style.name.startswith("Heading"):
                current_section = text
                full_text += f"\n[Section: {current_section}]\n"
            else:
                full_text += f"{text}\n"
        elif isinstance(block, Table):
            if not block.rows:
                continue
            headers = [cell.text.strip() for cell in block.rows[0].cells]
            table_text = "\n| " + " | ".join(headers) + " |\n"
            table_text += "| " + " | ".join(["---"] * len(headers)) + " |\n"
            for row in block.rows[1:]:
                cells = [cell.text.strip() for cell in row.cells]
                table_text += "| " + " | ".join(cells) + " |\n"
            full_text += f"{table_text}\n"

    if not full_text.strip():
        return None

    return Document.create(
        source="docx",
        title=uploaded_file.name,
        content=full_text.strip(),
        metadata={"file": uploaded_file.name}
    )
