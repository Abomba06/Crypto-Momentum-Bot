import math
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, Iterable, List, Optional

import requests


POSITIVE_TERMS = {
    "adoption": 0.8,
    "approval": 0.9,
    "bullish": 1.0,
    "buy": 0.5,
    "breakout": 0.8,
    "growth": 0.7,
    "gain": 0.6,
    "gains": 0.6,
    "high": 0.4,
    "launch": 0.5,
    "partnership": 0.7,
    "rally": 1.0,
    "record": 0.5,
    "rebound": 0.8,
    "surge": 1.0,
    "strong": 0.6,
    "upgrade": 0.7,
}

NEGATIVE_TERMS = {
    "ban": 1.0,
    "bearish": 1.0,
    "crash": 1.2,
    "crime": 0.8,
    "decline": 0.7,
    "drop": 0.7,
    "dump": 0.9,
    "fall": 0.6,
    "fraud": 1.0,
    "hack": 1.1,
    "investigation": 0.9,
    "lawsuit": 0.9,
    "liquidation": 0.9,
    "loss": 0.7,
    "plunge": 1.0,
    "risk": 0.5,
    "sell": 0.5,
    "slump": 0.9,
    "weak": 0.6,
}

SYMBOL_KEYWORDS = {
    "BTC/USD": ["bitcoin", "btc"],
    "ETH/USD": ["ethereum", "eth"],
    "SOL/USD": ["solana", "sol"],
    "DOGE/USD": ["dogecoin", "doge"],
    "AVAX/USD": ["avalanche", "avax"],
    "LINK/USD": ["chainlink", "link"],
    "ADA/USD": ["cardano", "ada"],
    "MATIC/USD": ["polygon", "matic", "pol"],
    "UNI/USD": ["uniswap", "uni"],
    "LTC/USD": ["litecoin", "ltc"],
    "XRP/USD": ["ripple", "xrp"],
}


@dataclass(frozen=True)
class SentimentItem:
    source: str
    title: str
    published_at: Optional[datetime]
    score: float


@dataclass(frozen=True)
class SentimentSnapshot:
    symbol: str
    score: float
    label: str
    sample_size: int
    source_counts: Dict[str, int]
    items: List[SentimentItem]
    updated_at: datetime


def score_text(text: str) -> float:
    words = [token.strip(".,:;!?()[]{}<>\"'").lower() for token in text.split()]
    total = 0.0
    matches = 0
    for word in words:
        if word in POSITIVE_TERMS:
            total += POSITIVE_TERMS[word]
            matches += 1
        if word in NEGATIVE_TERMS:
            total -= NEGATIVE_TERMS[word]
            matches += 1
    if matches == 0:
        return 0.0
    raw = total / matches
    return max(-1.0, min(1.0, raw))


def score_label(score: float, bullish: float, bearish: float) -> str:
    if score >= bullish:
        return "bullish"
    if score <= bearish:
        return "bearish"
    return "neutral"


def parse_pubdate(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def default_keywords(symbol: str) -> List[str]:
    if symbol in SYMBOL_KEYWORDS:
        return SYMBOL_KEYWORDS[symbol]
    base = symbol.split("/", 1)[0].lower()
    return [base]


def parse_keyword_map(raw: str) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for chunk in raw.split(";"):
        piece = chunk.strip()
        if not piece or "=" not in piece:
            continue
        symbol, keywords_raw = piece.split("=", 1)
        keywords = [item.strip().lower() for item in keywords_raw.split(",") if item.strip()]
        if keywords:
            mapping[symbol.strip().upper()] = keywords
    return mapping


class SentimentClient:
    def __init__(
        self,
        session: requests.Session,
        *,
        enabled: bool,
        mode: str,
        sources: Iterable[str],
        lookback_hours: int,
        min_items: int,
        bullish_threshold: float,
        bearish_threshold: float,
        cache_secs: int,
        keyword_map: Optional[Dict[str, List[str]]] = None,
        news_limit: int = 10,
        twitter_limit: int = 10,
        twitter_rss_url: str = "",
    ):
        self.session = session
        self.enabled = enabled
        self.mode = mode
        self.sources = {source.strip().lower() for source in sources if source.strip()}
        self.lookback_hours = lookback_hours
        self.min_items = min_items
        self.bullish_threshold = bullish_threshold
        self.bearish_threshold = bearish_threshold
        self.cache_secs = cache_secs
        self.keyword_map = keyword_map or {}
        self.news_limit = news_limit
        self.twitter_limit = twitter_limit
        self.twitter_rss_url = twitter_rss_url.strip()
        self._cache: Dict[str, SentimentSnapshot] = {}

    def is_active(self) -> bool:
        return self.enabled and self.mode != "disabled" and bool(self.sources)

    def keywords_for(self, symbol: str) -> List[str]:
        return self.keyword_map.get(symbol.upper(), default_keywords(symbol))

    def get_sentiment(self, symbol: str, timeout: int) -> Optional[SentimentSnapshot]:
        if not self.is_active():
            return None

        cached = self._cache.get(symbol)
        if cached and (datetime.now(timezone.utc) - cached.updated_at).total_seconds() < self.cache_secs:
            return cached

        items: List[SentimentItem] = []
        if "news" in self.sources:
            items.extend(self._fetch_news(symbol, timeout))
        if "twitter" in self.sources and self.twitter_rss_url:
            items.extend(self._fetch_twitter(symbol, timeout))

        if len(items) < self.min_items:
            return None

        score = sum(item.score for item in items) / float(len(items))
        source_counts: Dict[str, int] = {}
        for item in items:
            source_counts[item.source] = source_counts.get(item.source, 0) + 1

        snapshot = SentimentSnapshot(
            symbol=symbol,
            score=max(-1.0, min(1.0, score)),
            label=score_label(score, self.bullish_threshold, self.bearish_threshold),
            sample_size=len(items),
            source_counts=source_counts,
            items=items,
            updated_at=datetime.now(timezone.utc),
        )
        self._cache[symbol] = snapshot
        return snapshot

    def _within_lookback(self, published_at: Optional[datetime]) -> bool:
        if published_at is None:
            return True
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours)
        return published_at >= cutoff

    def _fetch_rss_items(self, url: str, source: str, limit: int, timeout: int) -> List[SentimentItem]:
        response = self.session.get(url, timeout=timeout)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        items: List[SentimentItem] = []
        for node in root.findall(".//item"):
            title = (node.findtext("title") or "").strip()
            description = (node.findtext("description") or "").strip()
            published_at = parse_pubdate(node.findtext("pubDate"))
            if not self._within_lookback(published_at):
                continue
            combined = " ".join(part for part in [title, description] if part)
            if not combined:
                continue
            items.append(
                SentimentItem(
                    source=source,
                    title=title or combined[:120],
                    published_at=published_at,
                    score=score_text(combined),
                )
            )
            if len(items) >= limit:
                break
        return items

    def _fetch_news(self, symbol: str, timeout: int) -> List[SentimentItem]:
        query = " OR ".join(self.keywords_for(symbol))
        encoded_query = urllib.parse.quote(f"{query} when:{max(1, math.ceil(self.lookback_hours / 24.0))}d")
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        return self._fetch_rss_items(url, "news", self.news_limit, timeout)

    def _fetch_twitter(self, symbol: str, timeout: int) -> List[SentimentItem]:
        query = urllib.parse.quote(" OR ".join(self.keywords_for(symbol)))
        url = self.twitter_rss_url.format(query=query)
        return self._fetch_rss_items(url, "twitter", self.twitter_limit, timeout)
