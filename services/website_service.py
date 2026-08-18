import requests

from bs4 import BeautifulSoup
def extract_website_text(url):
    if not url.strip():
        return ""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/138.0 Safari/537.36"
        )
    }
    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
    except requests.RequestException:
        return ""
    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )
    for tag in soup(
        [
            "script",
            "style",
            "noscript",
            "header",
            "footer",
            "nav",
            "aside"
        ]
    ):
        tag.decompose()
    text = soup.get_text(
        separator=" ",
        strip=True
    )
    text = " ".join(
        text.split()
    )
    return text
