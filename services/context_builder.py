import re

# Maps content_type values to human-readable citation labels
CONTENT_TYPE_LABELS = {
    "OCR_TEXT": "Scanned Page",
    "visual_description": "Visual Content",
}

# Maps visual_type values to specific labels
VISUAL_TYPE_LABELS = {
    "chart": "Chart",
    "image": "Image",
    "diagram": "Diagram",
}


def build_context(documents: list[dict]) -> dict:
    context_sections = []
    sources = []

    for idx, doc in enumerate(documents, start=1):
        metadata = doc.get("metadata", {})
        source_type = metadata.get("source", "unknown").upper()
        content_type = metadata.get("content_type", "")
        visual_type = metadata.get("visual_type", "")

        # Determine title
        if source_type == "YOUTUBE":
            title = metadata.get("title", "YouTube Video")
        elif source_type in ["PDF", "DOCX", "PPTX", "XLSX", "CSV", "TXT", "MD", "IMAGE"]:
            title = metadata.get("file", "Document")
        elif source_type == "WEBSITE":
            title = metadata.get("url", "Website")
        elif source_type in ["NOTES", "NOTE"]:
            title = metadata.get("title", "Notes")
        else:
            title = metadata.get("file", metadata.get("title", "Document"))

        citation_parts = []

        # Source-specific citation building
        if source_type == "YOUTUBE":
            start_time = doc.get("start") or metadata.get("start")
            if start_time is not None:
                start_sec = int(start_time)
                citation_parts.append(f"Timestamp: {start_sec // 60:02}:{start_sec % 60:02}")

        elif source_type == "PDF":
            # Try metadata page_number first, then scan text for [Page X] tag
            page = metadata.get("page_number") or metadata.get("page")
            if page:
                citation_parts.append(f"Page: {page}")
            else:
                page_match = re.search(r"\[Page\s*(\d+)\]", doc.get("text", ""))
                if page_match:
                    citation_parts.append(f"Page: {page_match.group(1)}")

            # Add content type label for scanned pages and visual content
            if content_type == "OCR_TEXT":
                citation_parts.append("Scanned Page — OCR")
            elif content_type == "visual_description":
                visual_label = VISUAL_TYPE_LABELS.get(visual_type, "Visual")
                citation_parts.append(visual_label)

        elif source_type == "PPTX":
            slide_num = metadata.get("slide_number")
            if slide_num:
                citation_parts.append(f"Slide: {slide_num}")
            else:
                slide_match = re.search(r"\[Slide\s*(\d+)\]", doc.get("text", ""))
                if slide_match:
                    citation_parts.append(f"Slide: {slide_match.group(1)}")

            if content_type == "visual_description":
                visual_label = VISUAL_TYPE_LABELS.get(visual_type, "Visual")
                citation_parts.append(visual_label)

        elif source_type == "XLSX":
            sheet_match = re.search(r"\[Sheet:\s*([^\]]+)\]", doc.get("text", ""))
            if sheet_match:
                citation_parts.append(f"Sheet: {sheet_match.group(1)}")

        elif source_type in ["DOCX", "MD"]:
            section_match = re.search(r"\[Section:\s*([^\]]+)\]", doc.get("text", ""))
            if section_match:
                citation_parts.append(f"Section: {section_match.group(1)}")

        elif source_type == "IMAGE":
            if content_type == "OCR_TEXT":
                citation_parts.append("OCR Text")
            elif content_type == "visual_description":
                citation_parts.append("Image Analysis")

        elif source_type == "WEBSITE":
            citation_parts.append("Web Link")

        elif source_type in ["NOTES", "NOTE"]:
            citation_parts.append("User Notes")

        citation_str = " | ".join(citation_parts) if citation_parts else ""

        section = f"""[Document #{idx}]
Source   : {source_type}
Title    : {title}
{f'Citation : {citation_str}' if citation_str else ''}
Content  :
{doc["text"].strip()}
-------------------------"""
        context_sections.append(section)

        if title not in sources:
            sources.append(title)

    return {
        "context": "\n\n".join(context_sections),
        "sources": sources
    }
