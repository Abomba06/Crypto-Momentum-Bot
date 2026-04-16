from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from app.sentiment import SentimentItem, SentimentSnapshot


@dataclass(frozen=True)
class ReplayEvent:
    timestamp: datetime
    symbol: str
    twitter_score: float = 0.0
    news_score: float = 0.0
    confirmation_state: str = "unconfirmed"
    dominant_event_type: str = "none"
    acceleration: float = 0.0
    author: str = "replay"
    text: str = ""


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def event_to_snapshot(event: ReplayEvent) -> SentimentSnapshot:
    ts = _ensure_utc(event.timestamp)
    twitter_item = SentimentItem(
        source="twitter",
        source_name=event.author,
        title=event.text or event.dominant_event_type,
        published_at=ts,
        score=event.twitter_score,
        weight=max(0.5, abs(event.twitter_score) + 0.5),
        relevance=1.0,
        event_tags=[] if event.dominant_event_type == "none" else [event.dominant_event_type],
        author_display_name=event.author,
    )
    items = [twitter_item]
    top_twitter_posts = [twitter_item.title]
    top_news_headlines: List[str] = []
    source_counts: Dict[str, int] = {"twitter": 1}
    score = event.twitter_score
    if abs(event.news_score) > 0:
        news_item = SentimentItem(
            source="news",
            source_name="replay_news",
            title=f"{event.dominant_event_type} confirmation",
            published_at=ts,
            score=event.news_score,
            weight=max(0.5, abs(event.news_score) + 0.5),
            relevance=1.0,
            event_tags=[] if event.dominant_event_type == "none" else [event.dominant_event_type],
        )
        items.append(news_item)
        top_news_headlines.append(news_item.title)
        source_counts["news"] = 1
        score = (event.twitter_score * 0.8) + (event.news_score * 0.2)
    label = "bullish" if score >= 0.15 else "bearish" if score <= -0.15 else "neutral"
    return SentimentSnapshot(
        symbol=event.symbol,
        score=score,
        label=label,
        sample_size=len(items),
        source_counts=source_counts,
        items=items,
        top_headlines=[item.title for item in items[:2]],
        event_counts={} if event.dominant_event_type == "none" else {event.dominant_event_type: 1},
        acceleration=event.acceleration,
        updated_at=ts,
        top_twitter_posts=top_twitter_posts,
        top_news_headlines=top_news_headlines,
        primary_twitter_score=event.twitter_score,
        news_confirmation_score=event.news_score,
        confirmation_state=event.confirmation_state,
        dominant_event_type=event.dominant_event_type,
        predicted_event_type=event.dominant_event_type,
        predicted_event_probability=0.7 if event.dominant_event_type != "none" else 0.0,
    )


def active_event_snapshot(
    event_stream: Dict[str, List[ReplayEvent]],
    symbol: str,
    ts: datetime,
    max_age_minutes: int,
) -> Optional[SentimentSnapshot]:
    events = event_stream.get(symbol, [])
    if not events:
        return None
    current_ts = _ensure_utc(ts)
    eligible = [
        event
        for event in events
        if 0 <= (current_ts - _ensure_utc(event.timestamp)).total_seconds() <= max_age_minutes * 60
    ]
    if not eligible:
        return None
    best = max(eligible, key=lambda item: abs(item.twitter_score) + (0.35 * abs(item.news_score)))
    return event_to_snapshot(best)
