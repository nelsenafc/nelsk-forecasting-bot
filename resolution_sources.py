"""Read the pages a question says it resolves on.

Most tournament questions name the exact page that settles them: a tracker, a
statistics release, a court docket. The web-search model often cannot open those
pages, because many render with JavaScript or turn scripts away. In the bot's own
research on 37 closed questions, 19 said they could not open or verify something.
This module reads the named pages through a rendering reader [r.jina.ai, free, sent
public URLs only] and falls back to a plain fetch, so the forecasters see what the
resolving page shows today.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from urllib.parse import urlparse

import requests

READER = "https://r.jina.ai/"
URL = re.compile(r"https?://[^\s)\]>\"'<]+")
SKIP_DOMAINS = ("metaculus.com",)
MAX_SOURCES = 2
MAX_CHARS = 12000
MIN_CHARS = 200  # less than this is an empty shell: a page that needs JavaScript, or a block page
BROWSER = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
}


def source_urls(question) -> list[str]:
    """URLs named in the resolution criteria and fine print, in order, leaving out Metaculus links."""
    text = f"{question.resolution_criteria or ''}\n{question.fine_print or ''}"
    found: list[str] = []
    for url in URL.findall(text):
        url = url.rstrip(".,;:")
        if url in found or any(domain in urlparse(url).netloc for domain in SKIP_DOMAINS):
            continue
        found.append(url)
    return found[:MAX_SOURCES]


class _VisibleText(HTMLParser):
    HIDDEN = {"script", "style", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs) -> None:
        if tag in self.HIDDEN:
            self.hidden += 1

    def handle_endtag(self, tag) -> None:
        if tag in self.HIDDEN and self.hidden:
            self.hidden -= 1

    def handle_data(self, data) -> None:
        if not self.hidden and data.strip():
            self.parts.append(data.strip())


def _through_reader(url: str) -> str:
    response = requests.get(READER + url, headers={"X-Return-Format": "text"}, timeout=45)
    response.raise_for_status()
    return response.text


def _direct(url: str) -> str:
    response = requests.get(url, headers=BROWSER, timeout=20)
    response.raise_for_status()
    if "html" not in response.headers.get("content-type", ""):
        return ""
    parser = _VisibleText()
    parser.feed(response.text)
    return " ".join(parser.parts)


def read_page(url: str) -> str | None:
    """The page's text, rendered when possible, or None if neither route gets real content."""
    for fetch in (_through_reader, _direct):
        try:
            text = fetch(url).strip()
        except requests.RequestException:
            continue
        if len(text) >= MIN_CHARS:
            return text[:MAX_CHARS]
    return None


def read_sources(question) -> dict[str, str]:
    """Text of each readable resolution source, by URL."""
    urls = source_urls(question)
    if not urls:
        return {}
    with ThreadPoolExecutor(max_workers=len(urls)) as pool:
        texts = list(pool.map(read_page, urls))
    return {url: text for url, text in zip(urls, texts) if text}
