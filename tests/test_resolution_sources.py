from types import SimpleNamespace

import requests

import resolution_sources
from resolution_sources import read_page, source_urls

LONG = "Approval 41.2%, disapprove 55.0% as of Oct 12. " * 10


class Response:
    def __init__(self, text, content_type="text/html", status=200):
        self.text = text
        self.status = status
        self.headers = {"content-type": content_type}

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"{self.status}")


def test_source_urls_reads_the_criteria_in_order_and_skips_metaculus():
    question = SimpleNamespace(
        resolution_criteria=(
            "Resolves per [RealClearPolling](https://www.realclearpolling.com/polls/approval/trump). "
            "See https://www.metaculus.com/questions/1/ for context."
        ),
        fine_print="Backup source: https://www.bls.gov/news.release/empsit.nr0.htm. "
        "Again: https://www.realclearpolling.com/polls/approval/trump",
    )
    assert source_urls(question) == [
        "https://www.realclearpolling.com/polls/approval/trump",
        "https://www.bls.gov/news.release/empsit.nr0.htm",
    ]
    assert source_urls(SimpleNamespace(resolution_criteria=None, fine_print=None)) == []


def test_read_page_prefers_the_rendering_reader(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return Response(LONG, "text/plain")

    monkeypatch.setattr(resolution_sources.requests, "get", fake_get)
    assert read_page("https://example.gov/data") == LONG.strip()
    assert calls == ["https://r.jina.ai/https://example.gov/data"]


def test_read_page_falls_back_to_a_plain_fetch(monkeypatch):
    def fake_get(url, **kwargs):
        if url.startswith(resolution_sources.READER):
            raise requests.ConnectionError("reader down")
        return Response(f"<html><script>var x = 1;</script><body><p>{LONG}</p></body></html>")

    monkeypatch.setattr(resolution_sources.requests, "get", fake_get)
    text = read_page("https://example.gov/data")
    assert text.startswith("Approval 41.2%") and "var x" not in text


def test_read_page_gives_up_on_empty_shells(monkeypatch):
    def fake_get(url, **kwargs):
        if url.startswith(resolution_sources.READER):
            return Response("Loading...", "text/plain")
        return Response("<html><body><div id='app'></div></body></html>")

    monkeypatch.setattr(resolution_sources.requests, "get", fake_get)
    assert read_page("https://example.org/tracker") is None
