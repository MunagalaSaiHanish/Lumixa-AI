import re

def build_context(documents: list[dict]) -> dict:
    context_sections = []
    sources = []
    for idx, doc in enumerate(documents, start=1):
        metadata = doc.get("metadata", {})
        source_type = metadata.get("source", "unknown").upper()
        if source_type == "YOUTUBE":
            title = metadata.get("title", "YouTube Video")
        elif source_type in ["PDF", "DOCX", "PPTX", "XLSX", "CSV", "TXT", "MD"]:
            title = metadata.get("file", "Document")
        elif source_type == "WEBSITE":
            title = metadata.get("url", "Website")
        elif source_type == "NOTES":
            title = metadata.get("title", "Notes")
        else:
            title = "Document"

        citation_parts = []
        if source_type == "YOUTUBE":
            start_time = doc.get("start") or metadata.get("start")
            if start_time is not None:
                start_sec = int(start_time)
                citation_parts.append(f"Timestamp: {start_sec // 60:02}:{start_sec % 60:02}")
        elif source_type == "PDF":
            page = metadata.get("page")
            if page:
                citation_parts.append(f"Page: {page}")
            else:
                page_match = re.search(r"\[Page\s*(\d+)\]", doc["text"])
                if page_match:
                    citation_parts.append(f"Page: {page_match.group(1)}")
                else:
                    citation_parts.append("Page: Unknown")
        elif source_type == "PPTX":
            slide_match = re.search(r"\[Slide\s*(\d+)\]", doc["text"])
            if slide_match:
                citation_parts.append(f"Slide: {slide_match.group(1)}")
        elif source_type == "XLSX":
            sheet_match = re.search(r"\[Sheet:\s*([^\]]+)\]", doc["text"])
            if sheet_match:
                citation_parts.append(f"Sheet: {sheet_match.group(1)}")
        elif source_type in ["DOCX", "MD"]:
            section_match = re.search(r"\[Section:\s*([^\]]+)\]", doc["text"])
            if section_match:
                citation_parts.append(f"Section: {section_match.group(1)}")
        elif source_type == "WEBSITE":
            citation_parts.append("Web Link")
        elif source_type == "NOTES":
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
