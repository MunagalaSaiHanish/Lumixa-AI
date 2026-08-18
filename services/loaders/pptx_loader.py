import io
from pptx import Presentation
from pptx.util import Inches
from pptx.enum.shapes import MSO_SHAPE_TYPE
from services.models.document import Document
from services.vision_service import analyze_image
from services.ocr_service import ocr_image
from config import ENABLE_VISION_PROCESSING


def _extract_shape_image_bytes(shape) -> bytes:
    """
    Try to extract raw image bytes from a picture shape.
    Returns empty bytes on failure.
    """
    try:
        return shape.image.blob
    except Exception:
        return b""


def _render_chart_to_image(shape) -> bytes:
    """
    PPTX charts are not easily rendered without external tools.
    Instead, we extract the chart's underlying data as text,
    which is more reliable than rendering to an image.
    Returns a text description of the chart data, or empty string.
    """
    try:
        chart = shape.chart
        chart_type = str(chart.chart_type)
        title = ""
        if chart.has_title and chart.chart_title.has_text_frame:
            title = chart.chart_title.text_frame.text.strip()

        series_info = []
        for series in chart.series:
            series_name = series.name or "Series"
            try:
                values = [v for v in series.values if v is not None]
                series_info.append(f"Series '{series_name}': {values}")
            except Exception:
                series_info.append(f"Series '{series_name}': (values not extractable)")

        description_parts = []
        if title:
            description_parts.append(f"Chart Title: {title}")
        description_parts.append(f"Chart Type: {chart_type}")
        for s in series_info:
            description_parts.append(s)

        return "\n".join(description_parts)
    except Exception as e:
        print(f"[PPTX] Chart data extraction failed: {e}")
        return ""


def load_pptx(uploaded_file) -> list:
    """
    Phase 3 upgraded PPTX loader.

    Returns a list of Document objects:
    - One document containing all slide text (same as Phase 2)
    - Additional documents for chart data and image descriptions per slide

    Visual processing only runs when ENABLE_VISION_PROCESSING = True.
    """
    try:
        prs = Presentation(uploaded_file)
    except Exception:
        return []

    source_name = uploaded_file.name
    documents = []

    # --- Pass 1: Extract all slide text (same as Phase 2) ---
    full_text = ""
    for idx, slide in enumerate(prs.slides, start=1):
        slide_text = f"[Slide {idx}]\n"
        if slide.shapes.title:
            title_text = slide.shapes.title.text.strip()
            if title_text:
                slide_text += f"Title: {title_text}\n"

        texts = []
        for shape in slide.shapes:
            if shape == slide.shapes.title:
                continue
            if shape.has_text_frame:
                text = shape.text.strip()
                if text:
                    texts.append(text)

        if texts:
            slide_text += "\n".join(texts) + "\n"

        if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                slide_text += f"\nSpeaker Notes:\n{notes}\n"

        full_text += slide_text + "\n"

    if full_text.strip():
        doc = Document.create(
            source="pptx",
            title=source_name,
            content=full_text.strip(),
            metadata={"file": source_name, "source": "pptx"}
        )
        documents.append(doc)

    if not ENABLE_VISION_PROCESSING:
        return documents

    # --- Pass 2: Extract chart data and images per slide ---
    for slide_idx, slide in enumerate(prs.slides, start=1):
        slide_title = ""
        if slide.shapes.title:
            slide_title = slide.shapes.title.text.strip()

        image_index = 0
        for shape in slide.shapes:
            # Handle chart shapes: extract data as text
            if shape.has_chart:
                print(f"[PPTX] Extracting chart data from slide {slide_idx}")
                chart_description = _render_chart_to_image(shape)
                if chart_description:
                    content = (
                        f"[Slide {slide_idx}] [Chart Data]\n"
                        f"{chart_description}"
                    )
                    doc = Document.create(
                        source="pptx",
                        title=source_name,
                        content=content,
                        metadata={
                            "file": source_name,
                            "source": "pptx",
                            "slide_number": slide_idx,
                            "slide_title": slide_title,
                            "content_type": "visual_description",
                            "visual_type": "chart"
                        }
                    )
                    documents.append(doc)

            # Handle picture shapes: send to vision model
            elif shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                image_index += 1
                img_bytes = _extract_shape_image_bytes(shape)
                if not img_bytes:
                    continue
                print(f"[PPTX] Analyzing image {image_index} on slide {slide_idx}")
                description = analyze_image(
                    image_bytes=img_bytes,
                    context_hint=source_name,
                    is_chart=False,
                    slide_number=slide_idx
                )
                if description:
                    content = (
                        f"[Slide {slide_idx}] [Visual Content — Image {image_index}]\n"
                        f"{description}"
                    )
                    doc = Document.create(
                        source="pptx",
                        title=source_name,
                        content=content,
                        metadata={
                            "file": source_name,
                            "source": "pptx",
                            "slide_number": slide_idx,
                            "slide_title": slide_title,
                            "visual_index": image_index,
                            "content_type": "visual_description",
                            "visual_type": "image"
                        }
                    )
                    documents.append(doc)

    return documents
