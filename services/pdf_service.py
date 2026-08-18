import io
import fitz  # PyMuPDF
from typing import List, Dict, Any
from config import ENABLE_VISION_PROCESSING, VISION_MIN_TEXT_THRESHOLD


def extract_pdf_text(pdf_file) -> str:
    """
    Original Phase 2 text extraction — unchanged.
    Extracts machine-readable text page by page with [Page X] tags.
    """
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


def extract_pdf_pages(pdf_file) -> List[Dict[str, Any]]:
    """
    Phase 3 upgraded extractor.
    Returns a list of page records, each containing:
      - page_number
      - text (machine-readable text, may be empty for scanned pages)
      - images (list of image bytes extracted from this page)
      - is_scanned (True if text is too short to be useful)

    This is used by the Phase 3 visual intelligence pipeline.
    """
    try:
        pdf_bytes = pdf_file.read()
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return []

    pages = []
    for page_index in range(document.page_count):
        page = document.load_page(page_index)
        page_number = page_index + 1
        text = page.get_text().strip()
        is_scanned = len(text) < VISION_MIN_TEXT_THRESHOLD

        # Extract embedded images from this page
        embedded_images = []
        if ENABLE_VISION_PROCESSING:
            try:
                image_list = page.get_images(full=True)
                for img_info in image_list:
                    xref = img_info[0]
                    try:
                        base_image = document.extract_image(xref)
                        img_bytes = base_image["image"]
                        embedded_images.append(img_bytes)
                    except Exception as e:
                        print(f"[PDF] Failed to extract image xref={xref} on page {page_number}: {e}")
            except Exception as e:
                print(f"[PDF] Failed to get image list on page {page_number}: {e}")

        # For scanned pages: render the page as an image for OCR
        rendered_image = None
        if is_scanned and ENABLE_VISION_PROCESSING:
            try:
                # Render at 150 DPI — good enough for OCR, not wasteful
                mat = fitz.Matrix(150 / 72, 150 / 72)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                rendered_image = pix.tobytes("png")
            except Exception as e:
                print(f"[PDF] Failed to render page {page_number} for OCR: {e}")

        pages.append({
            "page_number": page_number,
            "text": text,
            "is_scanned": is_scanned,
            "embedded_images": embedded_images,
            "rendered_image": rendered_image,
        })

    document.close()
    return pages
