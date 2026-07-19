import fitz

def extract_pdf_text(pdf_file) -> str:
    try:
        document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    except Exception:
        return ""
    full_text = ""
    for page_number in range(document.page_count):
        text = document.load_page(page_number).get_text().strip()
        if text:
            full_text += f"[Page {page_number + 1}]\n{text}\n\n"
    document.close()
    return full_text.strip()
