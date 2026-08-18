import io
from typing import List
from PIL import Image
from services.models.document import Document
from services.ocr_service import ocr_image
from services.vision_service import analyze_image

# Supported image extensions
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _validate_image(image_bytes: bytes, filename: str) -> bool:
    """Check that the file is a valid image we can process."""
    ext = "." + filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        print(f"[Image] Unsupported file type: {ext}")
        return False
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  # Checks integrity without fully loading
        return True
    except Exception as e:
        print(f"[Image] Invalid image file {filename}: {e}")
        return False


def load_image(uploaded_file) -> List[Document]:
    """
    Load an image file (PNG, JPG, JPEG, WEBP) and create knowledge documents.

    Returns a list of Document objects:
    - One document with OCR-extracted text (if any text is visible)
    - One document with vision model description of the visual content

    This lets users ask questions like:
      "What does the chart show?"
      "Extract the text from this image."
      "Describe this diagram."
    """
    source_name = uploaded_file.name
    documents = []

    try:
        image_bytes = uploaded_file.read()
    except Exception as e:
        print(f"[Image] Failed to read file {source_name}: {e}")
        return []

    if not _validate_image(image_bytes, source_name):
        return []

    print(f"[Image] Processing image: {source_name} ({len(image_bytes)//1024}KB)")

    # --- Step 1: Run OCR to extract any visible text ---
    ocr_text = ocr_image(image_bytes)
    if ocr_text:
        doc = Document.create(
            source="image",
            title=source_name,
            content=f"[Image OCR Text]\n{ocr_text}",
            metadata={
                "file": source_name,
                "source": "image",
                "content_type": "OCR_TEXT"
            }
        )
        documents.append(doc)
        print(f"[Image] OCR extracted {len(ocr_text)} chars from {source_name}")

    # --- Step 2: Analyze with vision model for visual understanding ---
    # Guess if this is a chart based on filename keywords
    is_chart_hint = any(
        keyword in source_name.lower()
        for keyword in ["chart", "graph", "plot", "bar", "pie", "line", "sales", "revenue", "data"]
    )
    description = analyze_image(
        image_bytes=image_bytes,
        context_hint=source_name,
        is_chart=is_chart_hint
    )
    if description:
        doc = Document.create(
            source="image",
            title=source_name,
            content=f"[Image Visual Description]\n{description}",
            metadata={
                "file": source_name,
                "source": "image",
                "content_type": "visual_description",
                "visual_type": "image"
            }
        )
        documents.append(doc)
        print(f"[Image] Vision analysis completed for {source_name}")

    if not documents:
        print(f"[Image] No knowledge extracted from {source_name}")

    return documents
