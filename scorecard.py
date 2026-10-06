"""Score the bot's resolved tournament forecasts, and each part of the bot on its own.

Answers the questions that decide spending:
- Is it beating the field? Spot peer score, the prize metric, per tournament.
- Is it calibrated? Binary forecasts grouped by probability against how often the events happened.
- Which lab earns its cost? Each lab's own forecast, read from the bot's comment, scored on the
  same questions as the labs combined, plus the combination without each lab. Paired like
  this, it turns useful long before the peer score does.
- Does the price model help? Price model alone against the AI ensemble alone, on the price
  questions where both ran.

Resolved questions only, so the report is safe to print anywhere. No AI calls, no cost.
Resolved questions are cached in .cache/scorecard.json, since they never change.

Usage: python scorecard.py [--json scorecard.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone
from statistics import mean, stdev

import dotenv

dotenv.load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from forecasting_tools import MetaculusClient
from forecasting_tools.data_models.questions import QuestionState

from forecast_log import (
    MODEL_LINE,
    SUMMARY_SECTION,
    bot_comment_index,
    comment_text,
    tournament_name,
    type_name,
)

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "scorecard.json")
CACHE_VERSION = 1  # bump when build_record changes, so cached questions are read again
UNSCORED = {"annulled", "ambiguous"}
QUANTILES = (0.1, 0.5, 0.9)
Z80 = 1.2816  # normal quantile for an 80% interval
CALIBRATION_BINS = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
NUMBER = r"-?[\d,]*\.?\d+(?:e[+-]?\d+)?"
DATE = r"\d{4}-\d{2}-\d{2}"
VALUE = rf"(?:{DATE}|{NUMBER})"
RANGE = re.compile(rf"median ({VALUE}) \[80% range ({VALUE}) to ({VALUE})\]")
PERCENT = re.compile(r"^(\d+(?:\.\d+)?)%$")
FINAL_LINE = re.compile(r"^Final forecast: (.+)$", re.M)


def family(kind: str) -> str:
    return {"Binary": "binary", "Multiple choice": "choice"}.get(kind, "range")


# Reading the bot's comment ------------------------------------------------------------


def _value(text: str) -> float:
    if re.fullmatch(DATE, text):
        return datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp()
    return float(text.replace(",", ""))


def parse_forecast(text: str, kind: str, options: list[str] | None = None):
    """A forecast as the comment prints it: '6%', 'A 60% / B 40%' or 'median X [80% range A to B]'.

    Returns a probability, a dict of option probabilities, or [10th, 50th, 90th percentile].
    """
    text = text.strip()
    if family(kind) == "binary":
        match = PERCENT.match(text)
        return float(match.group(1)) / 100 if match else None
    if family(kind) == "choice":
        probabilities = {}
        for option in options or []:
            match = re.search(rf"(?:^|/ ){re.escape(option)} (\d+(?:\.\d+)?)%", text)
            if match:
                probabilities[option] = float(match.group(1)) / 100
        return probabilities if options and len(probabilities) == len(options) else None
    match = RANGE.search(text)
    if not match:
        return None
    median, low, high = (_value(group) for group in match.groups())
    return [low, median, high]


def comment_forecasts(comment: str, kind: str, options: list[str] | None = None) -> dict:
    """The final forecast, every model's forecast by lab, and the price-model parts."""
    section = SUMMARY_SECTION.search(comment or "")
    found = {"final": None, "labs": {}, "price_model": None, "ai_alone": None}
    if not section:
        return found
    block = section.group(1)
    final = FINAL_LINE.search(block)
    if final:
        found["final"] = parse_forecast(final.group(1), kind, options)
    for model, value in MODEL_LINE.findall(block):
        provider = model.split("/")[0]
        if provider == "stats":
            found["price_model"] = parse_forecast(value, kind)
            alone = value.split("AI ensemble alone:", 1)
            found["ai_alone"] = parse_forecast(alone[1], kind) if len(alone) == 2 else None
            continue
        parsed = parse_forecast(value, kind, options)
        if parsed is not None:
            found["labs"].setdefault(provider, []).append(parsed)
    return found


def combine(forecasts: list, kind: str):
    """Several forecasts into one, the way the ensemble combines labs.

    Binary in log-odds, multiple choice as an average, numeric by averaging percentiles
    [the bot mixes full CDFs, which this approximates].
    """
    if family(kind) == "binary":
        clipped = [min(max(p, 0.01), 0.99) for p in forecasts]
        return 1 / (1 + math.exp(-mean(math.log(p / (1 - p)) for p in clipped)))
    if family(kind) == "choice":
        return {option: mean(f[option] for f in forecasts) for option in forecasts[0]}
    return [mean(f[i] for f in forecasts) for i in range(len(QUANTILES))]


# Scoring --------------------------------------------------------------------------------


def log_score(forecast, record: dict) -> float:
    """100 x natural log of the probability given to what happened, as Metaculus scores. Higher is better."""
    if family(record["type"]) == "binary":
        p = min(max(forecast, 0.01), 0.99)
        return 100 * math.log(p if record["resolution"] == "yes" else 1 - p)
    floored = {option: max(p, 0.005) for option, p in forecast.items()}
    return 100 * math.log(floored[record["resolution"]] / sum(floored.values()))


def range_loss(forecast, record: dict) -> float:
    """Pinball loss at the 10th, 50th and 90th percentiles, as % of the question's range. Lower is better."""
    outcome = record["resolution"]
    losses = [
        q * (outcome - x) if outcome >= x else (1 - q) * (x - outcome) for q, x in zip(QUANTILES, forecast)
    ]
    return 100 * mean(losses) / (record["upper"] - record["lower"])


def gain(forecast, reference, record: dict) -> float:
    """How much better a forecast scored than a reference on one question. + is better."""
    if family(record["type"]) == "range":
        return range_loss(reference, record) - range_loss(forecast, record)
    return log_score(forecast, record) - log_score(reference, record)


def summary(values: list[float]) -> tuple[str, str]:
    """Mean and 80% interval, formatted."""
    if not values:
        return "-", "-"
    average = mean(values)
    if len(values) < 2:
        return f"{average:+.1f}", "-"
    half = Z80 * stdev(values) / math.sqrt(len(values))
    return f"{average:+.1f}", f"{average - half:+.1f} to {average + half:+.1f}"


# Collecting resolved questions -----------------------------------------------------------


def _score(question, *keys: str) -> float | None:
    data = (question.api_json.get("question", {}).get("my_forecasts") or {}).get("score_data") or {}
    for key in keys:
        if data.get(key) is not None:
            return float(data[key])
    return None


def _timestamp(value) -> float:
    return value.timestamp() if isinstance(value, datetime) else float(value)


def _outcome(resolution: str, lower: float, upper: float) -> float | None:
    if resolution == "above_upper_bound":
        return upper
    if resolution == "below_lower_bound":
        return lower
    try:
        return float(resolution)
    except ValueError:
        try:
            return datetime.fromisoformat(resolution.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None


def build_record(question, comment: str, tournament: str) -> dict | None:
    """What the scorecard needs from one resolved question, or None if it can't be scored."""
    resolution = question.api_json.get("question", {}).get("resolution") or question.resolution_string
    if question.state != QuestionState.RESOLVED or resolution is None or str(resolution).lower() in UNSCORED:
        return None
    kind = type_name(question)
    resolved_at = question.actual_resolution_time or question.close_time
    record = {
        "post_id": question.id_of_post,
        "question": question.question_text,
        "url": question.page_url,
        "tournament": tournament,
        "type": kind,
        "resolved_on": resolved_at.date().isoformat() if resolved_at else None,
        "peer_score": _score(question, "spot_peer_score", "peer_score"),
        "baseline_score": _score(question, "spot_baseline_score", "baseline_score"),
    }
    options = getattr(question, "options", None)
    if family(kind) == "binary":
        record["resolution"] = str(resolution).lower()
    elif family(kind) == "choice":
        if str(resolution) not in (options or []):
            return None
        record["resolution"], record["options"] = str(resolution), options
    else:
        record["lower"] = _timestamp(question.lower_bound)
        record["upper"] = _timestamp(question.upper_bound)
        record["resolution"] = _outcome(str(resolution), record["lower"], record["upper"])
        if record["resolution"] is None:
            return None
    found = comment_forecasts(comment, kind, options)
    record["final"] = found["final"]
    record["labs"] = {lab: combine(samples, kind) for lab, samples in found["labs"].items()}
    record["price_model"], record["ai_alone"] = found["price_model"], found["ai_alone"]
    return record


def collect(client: MetaculusClient, headers: dict, cache: dict) -> tuple[list[dict], int]:
    """Scorable resolved questions, and how many questions could not be read."""
    records, failed = [], 0
    for post_id, comment in bot_comment_index(headers, client.get_current_user_id()).items():
        key = str(post_id)
        if key not in cache:
            try:
                question = client.get_question_by_post_id(post_id)
            except Exception:
                failed += 1
                continue
            tournament = tournament_name(question)
            if tournament is None or question.state != QuestionState.RESOLVED:
                continue
            cache[key] = build_record(question, comment_text(headers, comment), tournament) or {"skip": True}
        if not cache[key].get("skip"):
            records.append(cache[key])
    return records, failed


# The report ------------------------------------------------------------------------------


def field_section(records: list[dict]) -> list[str]:
    lines = [
        "## Against the field",
        "Spot peer score: above 0 beats the average forecaster on the question. Prize money goes with the "
        "square of a positive total.",
        "",
        "| Tournament | Questions | Total | Mean | 80% interval |",
        "|---|---|---|---|---|",
    ]
    groups = sorted({r["tournament"] for r in records}) + ["All"]
    for group in groups:
        scores = [r["peer_score"] for r in records if group in ("All", r["tournament"]) and r["peer_score"] is not None]
        average, spread = summary(scores)
        total = f"{sum(scores):+.1f}" if scores else "-"
        lines.append(f"| {group} | {len(scores)} | {total} | {average} | {spread} |")
    missing = sum(1 for r in records if r["peer_score"] is None)
    if missing:
        lines.append(f"\nPeer score missing on {missing} resolved question(s). Metaculus may not have scored them yet.")
    return lines


def calibration_section(records: list[dict]) -> list[str]:
    binary = [r for r in records if family(r["type"]) == "binary" and r["final"] is not None]
    if not binary:
        return []
    lines = [
        "## Calibration [binary questions, final forecast]",
        "",
        "| Forecast | Questions | Average forecast | Happened |",
        "|---|---|---|---|",
    ]
    for low, high in CALIBRATION_BINS:
        group = [r for r in binary if low <= r["final"] < high]
        if group:
            happened = sum(r["resolution"] == "yes" for r in group)
            average = mean(r["final"] for r in group)
            lines.append(
                f"| {low:.0%}-{min(high, 1):.0%} | {len(group)} | {average:.0%} | {happened} of {len(group)} |"
            )
    brier = mean((r["final"] - (r["resolution"] == "yes")) ** 2 for r in binary)
    lines.append(f"\nBrier score {brier:.3f}, lower is better. Always saying 50% scores 0.250.")
    return lines


def labs_section(records: list[dict]) -> list[str]:
    usable = [r for r in records if len(r["labs"]) >= 2]
    labs = sorted({lab for r in usable for lab in r["labs"]})
    if not labs:
        return []
    lines = [
        "## Which labs earn their cost",
        "Each row compares against all labs combined, on the same questions. + means better than all labs "
        "combined. A lab earns its cost when the combination without it scores below 0.",
    ]
    for title, kinds in (
        ("Binary and multiple choice, log score x100 per question", {"binary", "choice"}),
        ("Numeric and date, quantile loss as % of the range", {"range"}),
    ):
        subset = [r for r in usable if family(r["type"]) in kinds]
        if not subset:
            continue
        lines += [
            "",
            f"**{title}**",
            "",
            "| Lab | Questions | Lab alone | 80% interval | All labs without it | 80% interval |",
            "|---|---|---|---|---|---|",
        ]
        for lab in labs:
            alone, without = [], []
            for r in subset:
                if lab not in r["labs"]:
                    continue
                everyone = combine(list(r["labs"].values()), r["type"])
                others = [forecast for name, forecast in r["labs"].items() if name != lab]
                alone.append(gain(r["labs"][lab], everyone, r))
                without.append(gain(combine(others, r["type"]), everyone, r))
            lines.append(f"| {lab} | {len(alone)} | {' | '.join(summary(alone))} | {' | '.join(summary(without))} |")
    return lines


def price_section(records: list[dict]) -> list[str]:
    priced = [r for r in records if r["price_model"] and r["ai_alone"] and r["final"]]
    if not priced:
        return []
    losses = {
        "Price model alone": [range_loss(r["price_model"], r) for r in priced],
        "AI ensemble alone": [range_loss(r["ai_alone"], r) for r in priced],
        "Final blend": [range_loss(r["final"], r) for r in priced],
    }
    lines = [
        "## Price model",
        f"{len(priced)} price question(s). Quantile loss as % of the range, lower is better.",
        "",
        "| Forecast | Mean loss |",
        "|---|---|",
    ]
    lines += [f"| {name} | {mean(values):.2f} |" for name, values in losses.items()]
    edge, spread = summary([gain(r["price_model"], r["ai_alone"], r) for r in priced])
    lines.append(f"\nPrice model against the AI ensemble: {edge} per question [80% interval {spread}], + favours the price model.")
    return lines


def report(records: list[dict], failed: int, today: str) -> str:
    lines = [f"# nelsk scorecard - {today}", ""]
    if not records:
        lines.append("No resolved tournament questions yet.")
    else:
        counts = Counter(r["type"] for r in records)
        lines.append(
            f"{len(records)} resolved question(s): " + ", ".join(f"{n} {kind.lower()}" for kind, n in counts.most_common()) + "."
        )
        unread = sum(1 for r in records if r["final"] is None)
        if unread:
            lines.append(f"The final forecast could not be read from the comment on {unread} of them.")
        for section in (field_section, calibration_section, labs_section, price_section):
            part = section(records)
            if part:
                lines += [""] + part
    if failed:
        lines.append(f"\n{failed} question(s) could not be fetched from Metaculus on this run.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", help="Also write the scored questions to this file")
    args = parser.parse_args()
    cache = {}
    if os.path.exists(CACHE):
        with open(CACHE) as file:
            saved = json.load(file)
        if saved.get("version") == CACHE_VERSION:
            cache = saved["questions"]
    client = MetaculusClient()
    headers = {"Authorization": f"Token {os.environ['METACULUS_TOKEN']}"}
    records, failed = collect(client, headers, cache)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as file:
        json.dump({"version": CACHE_VERSION, "questions": cache}, file)
    text = report(records, failed, datetime.now(timezone.utc).date().isoformat())
    print(text)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as file:
            file.write(text + "\n")
    if args.json:
        with open(args.json, "w") as file:
            json.dump(records, file, indent=1)


if __name__ == "__main__":
    main()
