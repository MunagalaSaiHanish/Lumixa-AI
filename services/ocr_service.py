import io
from typing import Optional
from PIL import Image

# Try to import pytesseract. If Tesseract is not installed on the system,
# OCR will gracefully fall back to returning empty text.
TESSERACT_AVAILABLE = False
try:
    import pytesseract
    # Also verify the binary is actually reachable
    pytesseract.get_tesseract_version()
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False
    print("[OCR] Tesseract binary not found. OCR is disabled. Install from: https://github.com/UB-Mannheim/tesseract/wiki")

def ocr_image(image_bytes: bytes) -> str:
    """
    Run OCR on an image and return extracted text.

    Uses pytesseract which requires the Tesseract binary installed:
    Windows: https://github.com/UB-Mannheim/tesseract/wiki
    Linux:   sudo apt-get install tesseract-ocr
    macOS:   brew install tesseract

    Returns empty string if Tesseract is unavailable or OCR fails.
    """
    if not TESSERACT_AVAILABLE:
        return ""
    if not image_bytes:
        return ""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"[OCR] OCR failed: {e}")
        return ""

def ocr_pil_image(img: Image.Image) -> str:
    """
    Run OCR directly on a PIL Image object.
    Used when we already have the image in memory (e.g. from PDF rendering).
    """
    if not TESSERACT_AVAILABLE:
        return ""
    try:
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"[OCR] OCR failed on PIL image: {e}")
        return ""
