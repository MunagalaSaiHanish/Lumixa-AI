import requests
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    InvalidVideoId,
    YouTubeRequestFailed,
    FailedToCreateConsentCookie
)

class TranscriptSegment:
    def __init__(self, text, start, duration):
        self.text = text
        self.start = start
        self.duration = duration

def get_transcript(video_id: str) -> list[TranscriptSegment]:
    """transcript - English subtitles and auto gen"""
    primary_error = None
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        selected_transcript = None
        try:
            selected_transcript = transcript_list.find_manually_created_transcript(['en'])
        except Exception:
            pass
        if not selected_transcript:
            try:
                selected_transcript = transcript_list.find_generated_transcript(['en'])
            except Exception:
                pass
        if not selected_transcript:
            for t in transcript_list:
                try:
                    selected_transcript = t.translate('en')
                    break
                except Exception:
                    pass
        if not selected_transcript:
            all_transcripts = list(transcript_list)
            if all_transcripts:
                selected_transcript = all_transcripts[0]
        if not selected_transcript:
            raise NoTranscriptFound(video_id, "en", transcript_list)
        raw_transcript = selected_transcript.fetch()
        segments = []
        for entry in raw_transcript:
            if hasattr(entry, 'text'):
                text = getattr(entry, 'text', '')
                start = getattr(entry, 'start', 0.0)
                duration = getattr(entry, 'duration', 0.0)
            elif isinstance(entry, dict):
                text = entry.get('text', '')
                start = entry.get('start', 0.0)
                duration = entry.get('duration', 0.0)
            else:
                text = str(entry)
                start = 0.0
                duration = 0.0
            segments.append(TranscriptSegment(text, start, duration))
        return segments
    except Exception as e:
        primary_error = e

    ytdlp_error = None
    try:
        ydl_opts = {
            'writeautomaticsub': True,
            'subtitlesformat': 'json3',
            'skip_download': True,
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            info = ydl.extract_info(video_url, download=False)
            subtitles = info.get('subtitles') or {}
            auto_captions = info.get('automatic_captions') or {}
            all_subs = {}
            all_subs.update(auto_captions)
            all_subs.update(subtitles)
            if not all_subs:
                raise ValueError("No subtitles found.")
            lang = None
            if 'en' in subtitles:
                lang = 'en'
            elif 'en' in auto_captions:
                lang = 'en'
            else:
                lang = list(all_subs.keys())[0]
            sub_info = all_subs[lang]
            url = None
            if isinstance(sub_info, dict):
                url = sub_info.get('url')
            elif isinstance(sub_info, list) and sub_info:
                url = sub_info[0].get('url')
            if not url:
                raise ValueError("No subtitle URL.")
            if 'fmt=json3' not in url and '&fmt=' not in url:
                url += '&fmt=json3'
            if lang != 'en':
                url += '&tlang=en'
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            events = data.get('events', [])
            segments = []
            for ev in events:
                start_ms = ev.get('tStartMs', 0)
                duration_ms = ev.get('dDurationMs', 0)
                segs = ev.get('segs', [])
                text = "".join([s.get('utf8', '') for s in segs]).strip()
                if not text or text == '\n':
                    continue
                segments.append(TranscriptSegment(text, start_ms / 1000.0, duration_ms / 1000.0))
            if not segments:
                raise ValueError("No segments parsed.")
            return segments
    except Exception as e:
        ytdlp_error = e

    raise RuntimeError(classify_errors(primary_error, ytdlp_error))

def classify_errors(primary_error: Exception, ytdlp_error: Exception) -> str:
    """Classify errors user-friendly"""
    if isinstance(primary_error, TranscriptsDisabled):
        return "Subtitles/transcripts are disabled for this video by the owner."
    if isinstance(primary_error, NoTranscriptFound):
        return "No subtitles or transcripts exist for this video in any language."
    if isinstance(primary_error, VideoUnavailable):
        return "This video is unavailable (it might be private, deleted, or blocked)."
    if isinstance(primary_error, InvalidVideoId):
        return "The YouTube Video URL contains an invalid video ID."
    if isinstance(primary_error, FailedToCreateConsentCookie):
        return "YouTube requests are blocked due to consent cookie requirements. Please try again later."
    if isinstance(primary_error, YouTubeRequestFailed):
        return "YouTube request failed. This might be due to API rate limiting or temporary blocking."
    ytdlp_str = str(ytdlp_error).lower() if ytdlp_error else ""
    primary_str = str(primary_error).lower() if primary_error else ""
    combined_str = f"{primary_str} | {ytdlp_str}"
    if "private video" in combined_str:
        return "This video is private and cannot be accessed."
    if "sign in to confirm your age" in combined_str or "confirm your age" in combined_str:
        return "This video is age-restricted and requires signing in to YouTube."
    if "country" in combined_str or "geoblock" in combined_str or "region" in combined_str:
        return "This video is region-restricted/geoblocked in your current location."
    if "this video is unavailable" in combined_str:
        return "This video is unavailable or has been deleted."
    if "connection" in combined_str or "network" in combined_str or "timeout" in combined_str:
        return "Network issue: Unable to connect to YouTube servers. Please check your internet connection."
    return f"Failed to retrieve subtitles: {str(primary_error) or str(ytdlp_error) or 'Unknown error'}"

def transcript_to_text(transcript: list[TranscriptSegment]) -> str:
    """Transcript to flat text"""
    return " ".join([s.text for s in transcript]).strip()

def transcript_with_timestamps(transcript: list[TranscriptSegment]) -> list[dict]:
    """Format segments to dict"""
    return [{"text": s.text, "start": s.start, "end": s.start + s.duration} for s in transcript]
