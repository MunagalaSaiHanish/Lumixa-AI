from typing import List
from services.pdf_service import extract_pdf_pages
from services.models.document import Document
from services.ocr_service import ocr_image
from services.vision_service import analyze_image
from config import ENABLE_VISION_PROCESSING


def load_pdf(uploaded_file) -> List[Document]:
    """
    Phase 3 upgraded PDF loader.

    Returns a list of Document objects instead of just one.
    Each document represents one piece of knowledge:
      - Text document: all machine-readable text (as before)
      - OCR document: text extracted from scanned pages
      - Visual document: description of charts/images

    If vision is disabled, this behaves like the original Phase 2 loader
    but still returns a list (for consistent calling code in app.py).
    """
    pages = extract_pdf_pages(uploaded_file)
    if not pages:
        return []

    source_name = uploaded_file.name
    documents = []

    # --- Collect all machine-readable text into one document ---
    text_parts = []
    for page in pages:
        if page["text"]:
            text_parts.append(f"[Page {page['page_number']}]\n{page['text']}")

    if text_parts:
        full_text = "\n\n".join(text_parts)
        doc = Document.create(
            source="pdf",
            title=source_name,
            content=full_text,
            metadata={"file": source_name, "source": "pdf"}
        )
        documents.append(doc)
    elif not ENABLE_VISION_PROCESSING:
        # No text and vision is off — return empty
        return []

    if not ENABLE_VISION_PROCESSING:
        return documents

    # --- Process scanned pages with OCR ---
    for page in pages:
        if page["is_scanned"] and page["rendered_image"]:
            print(f"[PDF] Page {page['page_number']} appears scanned — running OCR")
            ocr_text = ocr_image(page["rendered_image"])
            if ocr_text:
                content = f"[Page {page['page_number']}] [OCR Extracted Text]\n{ocr_text}"
                doc = Document.create(
                    source="pdf",
                    title=source_name,
                    content=content,
                    metadata={
                        "file": source_name,
                        "source": "pdf",
                        "page_number": page["page_number"],
                        "content_type": "OCR_TEXT"
                    }
                )
                documents.append(doc)
            else:
                print(f"[PDF] OCR returned empty for page {page['page_number']}")

    # --- Process embedded images/charts with vision model ---
    for page in pages:
        for img_index, img_bytes in enumerate(page["embedded_images"]):
            print(f"[PDF] Analyzing image {img_index + 1} on page {page['page_number']}")
            description = analyze_image(
                image_bytes=img_bytes,
                context_hint=source_name,
                is_chart=True,  # assume charts unless clearly a photo
                page_number=page["page_number"]
            )
            if description:
                content = (
                    f"[Page {page['page_number']}] [Visual Content — Image {img_index + 1}]\n"
                    f"{description}"
                )
                doc = Document.create(
                    source="pdf",
                    title=source_name,
                    content=content,
                    metadata={
                        "file": source_name,
                        "source": "pdf",
                        "page_number": page["page_number"],
                        "visual_index": img_index + 1,
                        "content_type": "visual_description",
                        "visual_type": "image"
                    }
                )
                documents.append(doc)

    return documents
