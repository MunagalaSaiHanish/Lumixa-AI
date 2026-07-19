from pptx import Presentation
from services.models.document import Document

def load_pptx(uploaded_file) -> Document:
    try:
        prs = Presentation(uploaded_file)
    except Exception:
        return None

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

    if not full_text.strip():
        return None

    return Document.create(
        source="pptx",
        title=uploaded_file.name,
        content=full_text.strip(),
        metadata={"file": uploaded_file.name}
    )
