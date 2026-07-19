from services.youtube_service import extract_video_id
from services.transcript_service import get_transcript, transcript_to_text
from services.youtube_metadata import get_video_metadata
from services.models.document import Document

def load_youtube(url: str) -> Document:
    video_id = extract_video_id(url)
    if video_id is None:
        raise ValueError("Invalid YouTube URL.")
    metadata = get_video_metadata(url)
    transcript = get_transcript(video_id)
    text = transcript_to_text(transcript)
    return Document.create(
        source="youtube",
        title=metadata.get("title", "Unknown Video"),
        content=text,
        metadata={
            "channel": metadata.get("channel", "Unknown Channel"),
            "thumbnail": metadata.get("thumbnail", ""),
            "url": url
        }
    )
