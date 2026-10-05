"""Prediction-market prices as a reference signal.

Metaculus's bot rules allow public forecasts from other platforms. The bot looks
for a market on the same event and hands its price to the forecasters as one
input among many. Every call is time-boxed and fails quietly: research must
never stall on a market lookup.

Polymarket is real money and usually the sharpest price. Manifold is mostly play
money, so it is labelled as such and should count for less.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

POLYMARKET_SEARCH_URL = "https://gamma-api.polymarket.com/public-search"
MANIFOLD_SEARCH_URL = "https://api.manifold.markets/v0/search-markets"
MANIFOLD_MARKET_URL = "https://api.manifold.markets/v0/market/{market_id}"
REQUEST_TIMEOUT_SECONDS = 15
MAX_OUTCOMES_SHOWN = 8
MAX_RULES_CHARS = 600


@dataclass
class MarketQuote:
    source: str
    title: str
    url: str
    outcomes: list[tuple[str, float]]
    volume: float | None = None
    close_time: str | None = None
    rules: str = ""
    real_money: bool = True
    traders: int | None = None

    def describe(self) -> str:
        shown = sorted(self.outcomes, key=lambda pair: -pair[1])[:MAX_OUTCOMES_SHOWN]
        prices = ", ".join(f"{name} {price:.0%}" for name, price in shown)
        details = [f"{self.source}{'' if self.real_money else ' [play money]'}: \"{self.title}\""]
        details.append(f"prices: {prices}")
        if self.volume is not None:
            unit = "$" if self.real_money else "M$"
            details.append(f"volume {unit}{self.volume:,.0f}")
        if self.traders:
            details.append(f"{self.traders} traders")
        if self.close_time:
            details.append(f"closes {self.close_time[:10]}")
        details.append(self.url)
        return " | ".join(details)


def _get_json(url: str, params: dict | None = None) -> object:
    response = requests.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers={"User-Agent": "metaculus-forecasting-bot/1.0"},
    )
    response.raise_for_status()
    return response.json()


def _as_list(value: object) -> list:
    """Polymarket returns outcome lists as JSON-encoded strings."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _as_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _ms_to_date(milliseconds: object) -> str | None:
    seconds = _as_float(milliseconds)
    if seconds is None:
        return None
    return datetime.fromtimestamp(seconds / 1000, tz=timezone.utc).date().isoformat()


def parse_polymarket_search(payload: object) -> list[MarketQuote]:
    if not isinstance(payload, dict):
        return []
    quotes: list[MarketQuote] = []
    for event in payload.get("events") or []:
        slug = event.get("slug")
        url = f"https://polymarket.com/event/{slug}" if slug else "https://polymarket.com"
        for market in event.get("markets") or []:
            if market.get("closed") or market.get("active") is False:
                continue
            names = _as_list(market.get("outcomes"))
            prices = [_as_float(price) for price in _as_list(market.get("outcomePrices"))]
            if not names or len(names) != len(prices) or None in prices:
                continue
            quotes.append(
                MarketQuote(
                    source="Polymarket",
                    title=market.get("question") or event.get("title") or "",
                    url=url,
                    outcomes=[(str(name), float(price)) for name, price in zip(names, prices)],
                    volume=_as_float(market.get("volume")),
                    close_time=market.get("endDate"),
                    rules=(market.get("description") or "")[:MAX_RULES_CHARS],
                )
            )
    return quotes


def parse_manifold_market(market: dict) -> MarketQuote | None:
    if market.get("isResolved"):
        return None
    outcome_type = market.get("outcomeType")
    if outcome_type == "BINARY" and market.get("probability") is not None:
        probability = float(market["probability"])
        outcomes = [("Yes", probability), ("No", 1.0 - probability)]
    elif outcome_type == "MULTIPLE_CHOICE" and market.get("answers"):
        outcomes = [
            (str(answer.get("text")), float(answer.get("probability") or 0.0))
            for answer in market["answers"]
        ]
    else:
        return None
    return MarketQuote(
        source="Manifold",
        title=market.get("question") or "",
        url=market.get("url") or "https://manifold.markets",
        outcomes=outcomes,
        volume=_as_float(market.get("volume")),
        close_time=_ms_to_date(market.get("closeTime")),
        rules=(market.get("textDescription") or "")[:MAX_RULES_CHARS],
        real_money=market.get("token") == "CASH",
        traders=market.get("uniqueBettorCount"),
    )


def search_polymarket(query: str, limit: int = 5) -> list[MarketQuote]:
    try:
        payload = _get_json(
            POLYMARKET_SEARCH_URL,
            {"q": query, "limit_per_type": limit, "events_status": "active"},
        )
    except (requests.RequestException, ValueError) as error:
        logger.warning(f"Polymarket search failed for '{query}': {error}")
        return []
    return parse_polymarket_search(payload)


def search_manifold(query: str, limit: int = 5, max_detail_fetches: int = 2) -> list[MarketQuote]:
    try:
        results = _get_json(
            MANIFOLD_SEARCH_URL, {"term": query, "limit": limit, "filter": "open"}
        )
    except (requests.RequestException, ValueError) as error:
        logger.warning(f"Manifold search failed for '{query}': {error}")
        return []
    quotes: list[MarketQuote] = []
    detail_fetches = 0
    for market in results if isinstance(results, list) else []:
        if market.get("outcomeType") == "MULTIPLE_CHOICE" and detail_fetches < max_detail_fetches:
            detail_fetches += 1
            try:
                market = _get_json(MANIFOLD_MARKET_URL.format(market_id=market["id"]))
            except (requests.RequestException, ValueError, KeyError) as error:
                logger.warning(f"Manifold market fetch failed: {error}")
                continue
        quote = parse_manifold_market(market)
        if quote:
            quotes.append(quote)
    return quotes


def search_all(queries: list[str], per_source_limit: int = 5) -> list[MarketQuote]:
    quotes: list[MarketQuote] = []
    seen: set[tuple[str, str]] = set()
    for query in queries:
        for quote in search_polymarket(query, per_source_limit) + search_manifold(query, per_source_limit):
            key = (quote.source, quote.title)
            if key not in seen:
                seen.add(key)
                quotes.append(quote)
    return quotes


def format_candidates(quotes: list[MarketQuote]) -> str:
    lines = []
    for index, quote in enumerate(quotes):
        lines.append(f"[{index}] {quote.describe()}")
        if quote.rules:
            lines.append(f"    rules: {quote.rules}")
    return "\n".join(lines)


def format_market_section(matches: list[tuple[MarketQuote, str, str]]) -> str:
    """matches: (quote, match level, note on how it differs from the question)."""
    if not matches:
        return ""
    lines = [
        "## Prediction markets [reference prices only]",
        "Markets judged to be about the same event. Check the stated differences before leaning on a price. "
        "Play-money markets are weaker evidence than real-money ones.",
    ]
    for quote, level, note in matches:
        lines.append(f"- {quote.describe()} | match: {level} - {note}")
    return "\n".join(lines)
