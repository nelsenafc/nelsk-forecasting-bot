import json

from forecaster import parse_queries
from markets import (
    MarketQuote,
    format_market_section,
    parse_manifold_market,
    parse_polymarket_search,
)


POLYMARKET_PAYLOAD = {
    "events": [
        {
            "slug": "bank-indonesia-november-decision",
            "title": "Bank Indonesia November decision",
            "markets": [
                {
                    "question": "Will Bank Indonesia cut rates in November 2026?",
                    "outcomes": json.dumps(["Yes", "No"]),
                    "outcomePrices": json.dumps(["0.34", "0.66"]),
                    "volume": "125000.5",
                    "endDate": "2026-11-20T00:00:00Z",
                    "description": "Resolves Yes if BI announces a cut at its November meeting.",
                    "active": True,
                    "closed": False,
                },
                {
                    "question": "Closed market",
                    "outcomes": json.dumps(["Yes", "No"]),
                    "outcomePrices": json.dumps(["1", "0"]),
                    "closed": True,
                },
                {
                    "question": "Broken market",
                    "outcomes": json.dumps(["Yes", "No"]),
                    "outcomePrices": json.dumps(["0.5"]),
                },
            ],
        }
    ]
}


def test_polymarket_parsing_keeps_only_open_wellformed_markets():
    quotes = parse_polymarket_search(POLYMARKET_PAYLOAD)
    assert len(quotes) == 1
    quote = quotes[0]
    assert quote.outcomes == [("Yes", 0.34), ("No", 0.66)]
    assert quote.url == "https://polymarket.com/event/bank-indonesia-november-decision"
    assert quote.volume == 125000.5
    assert quote.real_money


def test_polymarket_parsing_tolerates_junk():
    assert parse_polymarket_search(None) == []
    assert parse_polymarket_search({"events": None}) == []


def test_manifold_binary_and_play_money_flag():
    quote = parse_manifold_market(
        {
            "outcomeType": "BINARY",
            "probability": 0.25,
            "question": "Will X happen?",
            "url": "https://manifold.markets/u/x",
            "token": "MANA",
            "closeTime": 1798761540000,
            "uniqueBettorCount": 12,
        }
    )
    assert quote.outcomes == [("Yes", 0.25), ("No", 0.75)]
    assert not quote.real_money
    assert quote.close_time == "2026-12-31"
    assert "[play money]" in quote.describe()


def test_manifold_multiple_choice_and_resolved_markets():
    quote = parse_manifold_market(
        {
            "outcomeType": "MULTIPLE_CHOICE",
            "question": "How many cuts?",
            "answers": [{"text": "0", "probability": 0.7}, {"text": "1", "probability": 0.3}],
        }
    )
    assert quote.outcomes == [("0", 0.7), ("1", 0.3)]
    assert parse_manifold_market({"isResolved": True, "outcomeType": "BINARY", "probability": 1}) is None


def test_market_section_lists_matches_with_notes():
    quote = MarketQuote(source="Polymarket", title="Cut in Nov?", url="https://polymarket.com/event/x", outcomes=[("Yes", 0.34), ("No", 0.66)])
    section = format_market_section([(quote, "close", "Deadline is a week later.")])
    assert "Polymarket" in section and "Yes 34%" in section and "Deadline is a week later." in section
    assert format_market_section([]) == ""


def test_query_parsing_strips_list_markers_but_keeps_years():
    text = '1. Bank Indonesia rate cut 2026\n- "BI rate decision November"\n\n3) extra query'
    assert parse_queries(text) == ["Bank Indonesia rate cut 2026", "BI rate decision November"]
