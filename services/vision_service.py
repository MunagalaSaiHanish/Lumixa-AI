import os
import base64
import hashlib
import io
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from config import VISION_MODEL

load_dotenv()

# The same client used by llm_service.py — same OpenRouter connection
_client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    timeout=120
)

# Session-level cache: image_hash -> description string
# This ensures the same image is never analyzed twice in one session
_vision_cache: dict[str, str] = {}

def _image_to_base64(image_bytes: bytes) -> str:
    """Convert raw image bytes to a base64 string for the API."""
    return base64.b64encode(image_bytes).decode("utf-8")

def _get_image_hash(image_bytes: bytes) -> str:
    """Create a short hash to use as a cache key."""
    return hashlib.md5(image_bytes).hexdigest()

def _resize_if_large(image_bytes: bytes, max_pixels: int = 1_500_000) -> bytes:
    """
    Resize image if it is very large to reduce token cost.
    Most charts and diagrams don't need more than 1500x1000 resolution.
    """
    img = Image.open(io.BytesIO(image_bytes))
    width, height = img.size
    if width * height > max_pixels:
        ratio = (max_pixels / (width * height)) ** 0.5
        new_w = int(width * ratio)
        new_h = int(height * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    output = io.BytesIO()
    # Save as PNG to preserve quality
    img.convert("RGB").save(output, format="PNG")
    return output.getvalue()

def analyze_image(
    image_bytes: bytes,
    context_hint: str = "",
    is_chart: bool = False,
    page_number: Optional[int] = None,
    slide_number: Optional[int] = None,
) -> str:
    """
    Send an image to the vision model and return a factual text description.

    Args:
        image_bytes: Raw bytes of the image (PNG, JPG, etc.)
        context_hint: Optional hint about what the image contains (e.g. file name)
        is_chart: If True, use a more structured chart-analysis prompt
        page_number: Page number for logging context
        slide_number: Slide number for logging context

    Returns:
        A factual text description of the image content.
        Returns empty string on failure.
    """
    if not image_bytes:
        return ""

    # Check cache first — avoid paying for the same image twice
    image_hash = _get_image_hash(image_bytes)
    if image_hash in _vision_cache:
        print(f"[Vision] Cache hit for image hash {image_hash[:8]}")
        return _vision_cache[image_hash]

    # Resize if too large before sending
    try:
        image_bytes = _resize_if_large(image_bytes)
    except Exception as e:
        print(f"[Vision] Image resize failed: {e}. Using original size.")

    b64_image = _image_to_base64(image_bytes)

    location_note = ""
    if page_number is not None:
        location_note = f" (from page {page_number})"
    elif slide_number is not None:
        location_note = f" (from slide {slide_number})"

    if is_chart:
        prompt = f"""You are analyzing a chart or graph{location_note}{' from: ' + context_hint if context_hint else ''}.

Provide a factual, structured description. Include:
- Chart type (bar, line, pie, scatter, etc.)
- Title if visible
- Axis labels and units if visible
- Categories or legend items
- Key values if clearly readable
- The highest and lowest data points if identifiable
- The overall trend or main insight
- Any notable comparisons

IMPORTANT: Only state values you can clearly read. If values are unclear, describe the trend instead. Never guess or hallucinate numbers."""
    else:
        prompt = f"""You are analyzing an image{location_note}{' from: ' + context_hint if context_hint else ''}.

Provide a clear, factual description including:
- What the image shows (chart, diagram, photo, screenshot, etc.)
- Main components or elements visible
- Any text that is visible
- Relationships or flows if it is a diagram
- Key labels or annotations

Be factual and specific. Do not guess content that is not visible."""

    try:
        response = _client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            max_tokens=800,
            temperature=0.1
        )
        description = response.choices[0].message.content or ""
        description = description.strip()

        # Cache the result
        _vision_cache[image_hash] = description
        print(f"[Vision] Analyzed image ({len(image_bytes)//1024}KB) → {len(description)} chars")
        return description

    except Exception as e:
        print(f"[Vision] Vision model call failed: {e}")
        return ""

def clear_cache():
    """Clear the vision cache. Called when workspace is cleared."""
    _vision_cache.clear()
