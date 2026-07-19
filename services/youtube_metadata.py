from yt_dlp import YoutubeDL

def get_video_metadata(video_url):
    try:
        ydl_opts = {
            "quiet": True,
            "skip_download": True,
        }
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
        return {
            "source": "youtube",
            "title": info.get("title", ""),
            "channel": info.get("channel", ""),
            "thumbnail": info.get("thumbnail", ""),
            "url": video_url,
            "description": info.get("description", ""),
        }
    except Exception as e:
        print(e)
        return {
            "source": "youtube",
            "title": "Unknown Video",
            "channel": "Unknown Channel",
            "thumbnail": "",
            "url": video_url,
            "description": "",
        }
