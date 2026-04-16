import math
import re
import pathlib
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, Iterable, List, Optional, Sequence, Set

import requests

from app.event_models import TwitterEvent, TwitterWatchAccount
from app.twitter_watchlist import load_watchlist


POSITIVE_TERMS = {
    "adoption": 0.8,
    "approval": 0.9,
    "beat": 0.7,
    "breakout": 0.8,
    "bullish": 1.0,
    "buy": 0.5,
    "expands": 0.5,
    "gain": 0.6,
    "gains": 0.6,
    "growth": 0.7,
    "high": 0.4,
    "launch": 0.5,
    "partnership": 0.7,
    "rally": 1.0,
    "record": 0.5,
    "rebound": 0.8,
    "recovery": 0.7,
    "settlement": 0.4,
    "strong": 0.6,
    "surge": 1.0,
    "upgrade": 0.7,
}

NEGATIVE_TERMS = {
    "ban": 1.0,
    "bearish": 1.0,
    "crackdown": 1.0,
    "crash": 1.2,
    "crime": 0.8,
    "decline": 0.7,
    "delist": 1.1,
    "drop": 0.7,
    "dump": 0.9,
    "exploit": 1.1,
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
    "token unlock": 0.7,
    "weak": 0.6,
}

EVENT_TERMS = {
    "approval": "approval",
    "etf": "etf",
    "hack": "hack",
    "exploit": "exploit",
    "lawsuit": "lawsuit",
    "investigation": "investigation",
    "partnership": "partnership",
    "listing": "listing",
    "delist": "delisting",
    "upgrade": "upgrade",
    "token unlock": "unlock",
    "launch": "launch",
    "liquidation": "liquidation",
}

SOURCE_WEIGHTS = {
    "coindesk": 1.15,
    "cointelegraph": 1.05,
    "bitcoin magazine": 1.0,
    "decrypt": 1.05,
    "the block": 1.15,
    "bloomberg": 1.25,
    "reuters": 1.25,
    "cnbc": 1.1,
    "forbes": 1.0,
    "yahoo finance": 1.0,
    "marketwatch": 1.0,
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
    source_name: str
    title: str
    published_at: Optional[datetime]
    score: float
    weight: float
    relevance: float
    event_tags: List[str]
    author_username: str = ""
    author_display_name: str = ""
    author_category: str = ""
    author_priority: float = 1.0
    author_reliability: float = 1.0
    raw_payload: Optional[Dict[str, str]] = None


@dataclass(frozen=True)
class SentimentSnapshot:
    symbol: str
    score: float
    label: str
    sample_size: int
    source_counts: Dict[str, int]
    items: List[SentimentItem]
    top_headlines: List[str]
    event_counts: Dict[str, int]
    acceleration: float
    updated_at: datetime
    top_twitter_posts: List[str] = None
    top_news_headlines: List[str] = None
    primary_twitter_score: float = 0.0
    news_confirmation_score: float = 0.0
    confirmation_state: str = "no_twitter_event"
    dominant_event_type: str = "none"


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def score_text(text: str) -> float:
    lowered = text.lower()
    total = 0.0
    matches = 0
    words = tokenize(text)
    for word in words:
        if word in POSITIVE_TERMS:
            total += POSITIVE_TERMS[word]
            matches += 1
        if word in NEGATIVE_TERMS:
            total -= NEGATIVE_TERMS[word]
            matches += 1
    for phrase, value in POSITIVE_TERMS.items():
        if " " in phrase and phrase in lowered:
            total += value
            matches += 1
    for phrase, value in NEGATIVE_TERMS.items():
        if " " in phrase and phrase in lowered:
            total -= value
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


def extract_source_name(title: str, fallback: str) -> str:
    if " - " in title:
        return title.rsplit(" - ", 1)[-1].strip().lower()
    return fallback


def source_weight(source_name: str) -> float:
    lowered = source_name.lower()
    for key, value in SOURCE_WEIGHTS.items():
        if key in lowered:
            return value
    return 0.95 if source_name == "twitter" else 1.0


def author_weight(priority: float, reliability: float) -> float:
    return max(0.5, min(2.0, (0.55 * priority) + (0.45 * reliability)))


def normalize_username(value: str) -> str:
    return value.strip().lstrip("@").lower()


def event_tags(text: str) -> List[str]:
    lowered = text.lower()
    tags = {tag for phrase, tag in EVENT_TERMS.items() if phrase in lowered}
    return sorted(tags)


def dominant_event_type(items: Sequence[SentimentItem]) -> str:
    counts: Dict[str, int] = {}
    for item in items:
        for tag in item.event_tags:
            counts[tag] = counts.get(tag, 0) + 1
    if not counts:
        return "none"
    return max(counts.items(), key=lambda item: item[1])[0]


def novelty_key(title: str) -> str:
    words = tokenize(title)
    return " ".join(words[:12])


def relevance_score(text: str, keywords: Sequence[str]) -> float:
    lowered = text.lower()
    if not keywords:
        return 1.0
    hits = sum(1 for keyword in keywords if keyword.lower() in lowered)
    if hits == 0:
        return 0.55
    return min(1.25, 0.7 + (0.18 * hits))


def recency_weight(published_at: Optional[datetime], now: datetime, lookback_hours: int) -> float:
    if published_at is None:
        return 0.85
    age_hours = max(0.0, (now - published_at).total_seconds() / 3600.0)
    half_life = max(2.0, lookback_hours / 3.0)
    weight = math.exp(-math.log(2.0) * (age_hours / half_life))
    return max(0.35, min(1.15, weight))


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
        twitter_watchlist_path: str = "",
        twitter_account_rss_url: str = "https://nitter.net/{username}/rss",
        twitter_primary_enabled: bool = False,
        twitter_primary_mode: str = "confirm",
        twitter_event_min_score: float = 0.2,
        twitter_event_max_age_minutes: int = 120,
        twitter_require_confirmation_for_entry: bool = False,
        twitter_allow_exit_interrupt: bool = True,
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
        self.twitter_watchlist_path = twitter_watchlist_path.strip()
        self.twitter_account_rss_url = twitter_account_rss_url.strip()
        self.twitter_primary_enabled = twitter_primary_enabled
        self.twitter_primary_mode = twitter_primary_mode.strip().lower()
        self.twitter_event_min_score = twitter_event_min_score
        self.twitter_event_max_age_minutes = twitter_event_max_age_minutes
        self.twitter_require_confirmation_for_entry = twitter_require_confirmation_for_entry
        self.twitter_allow_exit_interrupt = twitter_allow_exit_interrupt
        self.watchlist: List[TwitterWatchAccount] = load_watchlist(self.twitter_watchlist_path)
        self._cache: Dict[str, SentimentSnapshot] = {}

    def is_active(self) -> bool:
        return self.enabled and self.mode != "disabled" and bool(self.sources)

    def keywords_for(self, symbol: str) -> List[str]:
        return self.keyword_map.get(symbol.upper(), default_keywords(symbol))

    def get_sentiment(self, symbol: str, timeout: int) -> Optional[SentimentSnapshot]:
        if not self.is_active():
            return None

        cached = self._cache.get(symbol)
        now = datetime.now(timezone.utc)
        if cached and (now - cached.updated_at).total_seconds() < self.cache_secs:
            return cached

        twitter_items: List[SentimentItem] = []
        news_items: List[SentimentItem] = []
        if "twitter" in self.sources:
            twitter_items.extend(self._fetch_twitter(symbol, timeout))
        if "news" in self.sources:
            news_items.extend(self._fetch_news(symbol, timeout))

        if self.twitter_primary_enabled and not twitter_items:
            return None
        items: List[SentimentItem] = list(twitter_items)
        if twitter_items:
            items.extend(news_items)
        elif not self.twitter_primary_enabled:
            items.extend(news_items)

        if len(items) < self.min_items and not twitter_items:
            return None

        weighted_total = 0.0
        weight_total = 0.0
        source_counts: Dict[str, int] = {}
        event_counts: Dict[str, int] = {}
        for item in items:
            weighted_total += item.score * item.weight
            weight_total += item.weight
            source_counts[item.source] = source_counts.get(item.source, 0) + 1
            for tag in item.event_tags:
                event_counts[tag] = event_counts.get(tag, 0) + 1

        twitter_total = sum(item.score * item.weight for item in twitter_items)
        twitter_weight = sum(item.weight for item in twitter_items)
        news_total = sum(item.score * item.weight for item in news_items)
        news_weight = sum(item.weight for item in news_items)
        primary_twitter_score = twitter_total / max(twitter_weight, 1e-9) if twitter_items else 0.0
        news_confirmation_score = news_total / max(news_weight, 1e-9) if news_items else 0.0
        confirmation_state = self._confirmation_state(twitter_items, news_items)
        if twitter_items:
            score = (primary_twitter_score * 0.8) + (news_confirmation_score * 0.2 if news_items else 0.0)
        else:
            score = news_confirmation_score
        ordered = sorted(items, key=lambda item: abs(item.score * item.weight), reverse=True)
        top_headlines = [item.title for item in ordered[:2]]
        top_twitter_posts = [item.title for item in sorted(twitter_items, key=lambda item: abs(item.score * item.weight), reverse=True)[:2]]
        top_news_headlines = [item.title for item in sorted(news_items, key=lambda item: abs(item.score * item.weight), reverse=True)[:2]]

        recent = [item for item in items if item.published_at and (now - item.published_at).total_seconds() <= 6 * 3600]
        older = [item for item in items if item not in recent]
        recent_score = sum(item.score * item.weight for item in recent) / max(sum(item.weight for item in recent), 1e-9) if recent else score
        older_score = sum(item.score * item.weight for item in older) / max(sum(item.weight for item in older), 1e-9) if older else score
        acceleration = recent_score - older_score

        snapshot = SentimentSnapshot(
            symbol=symbol,
            score=max(-1.0, min(1.0, score)),
            label=score_label(score, self.bullish_threshold, self.bearish_threshold),
            sample_size=len(items),
            source_counts=source_counts,
            items=items,
            top_headlines=top_headlines,
            top_twitter_posts=top_twitter_posts,
            top_news_headlines=top_news_headlines,
            event_counts=event_counts,
            acceleration=max(-1.0, min(1.0, acceleration)),
            primary_twitter_score=max(-1.0, min(1.0, primary_twitter_score)),
            news_confirmation_score=max(-1.0, min(1.0, news_confirmation_score)),
            confirmation_state=confirmation_state,
            dominant_event_type=dominant_event_type(items),
            updated_at=now,
        )
        self._cache[symbol] = snapshot
        return snapshot

    def _confirmation_state(self, twitter_items: Sequence[SentimentItem], news_items: Sequence[SentimentItem]) -> str:
        if not twitter_items:
            return "no_twitter_event"
        if not news_items:
            return "unconfirmed"
        twitter_tags = {tag for item in twitter_items for tag in item.event_tags}
        news_tags = {tag for item in news_items for tag in item.event_tags}
        if twitter_tags and news_tags and twitter_tags.intersection(news_tags):
            return "confirmed_by_news"
        return "partially_confirmed"

    def _within_lookback(self, published_at: Optional[datetime], now: datetime) -> bool:
        if published_at is None:
            return True
        cutoff = now - timedelta(hours=self.lookback_hours)
        return published_at >= cutoff

    def _fetch_rss_items(
        self,
        url: str,
        source: str,
        limit: int,
        timeout: int,
        keywords: Sequence[str],
    ) -> List[SentimentItem]:
        response = self.session.get(url, timeout=timeout)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        now = datetime.now(timezone.utc)
        seen_titles: Set[str] = set()
        items: List[SentimentItem] = []
        for node in root.findall(".//item"):
            title = (node.findtext("title") or "").strip()
            description = (node.findtext("description") or "").strip()
            published_at = parse_pubdate(node.findtext("pubDate"))
            if not self._within_lookback(published_at, now):
                continue
            combined = " ".join(part for part in [title, description] if part)
            if not combined:
                continue
            key = novelty_key(title or combined)
            if key in seen_titles:
                continue
            seen_titles.add(key)
            src_name = extract_source_name(title, source)
            relevance = relevance_score(combined, keywords)
            weight = source_weight(src_name) * recency_weight(published_at, now, self.lookback_hours) * relevance
            items.append(
                SentimentItem(
                    source=source,
                    source_name=src_name,
                    title=title or combined[:120],
                    published_at=published_at,
                    score=score_text(combined),
                    weight=weight,
                    relevance=relevance,
                    event_tags=event_tags(combined),
                )
            )
            if len(items) >= limit:
                break
        return items

    def _fetch_news(self, symbol: str, timeout: int) -> List[SentimentItem]:
        keywords = self.keywords_for(symbol)
        query = " OR ".join(keywords)
        encoded_query = urllib.parse.quote(f"{query} when:{max(1, math.ceil(self.lookback_hours / 24.0))}d")
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        return self._fetch_rss_items(url, "news", self.news_limit, timeout, keywords)

    def _fetch_twitter(self, symbol: str, timeout: int) -> List[SentimentItem]:
        keywords = self.keywords_for(symbol)
        if self.watchlist:
            return self._fetch_watchlist_twitter(symbol, timeout, keywords)
        query = urllib.parse.quote(" OR ".join(keywords))
        url = self.twitter_rss_url.format(query=query)
        items = self._fetch_rss_items(url, "twitter", self.twitter_limit, timeout, keywords)
        return [item for item in items if item.weight >= self.twitter_event_min_score]

    def _fetch_watchlist_twitter(self, symbol: str, timeout: int, keywords: Sequence[str]) -> List[SentimentItem]:
        now = datetime.now(timezone.utc)
        items: List[SentimentItem] = []
        for account in self.watchlist:
            url = self.twitter_account_rss_url.format(username=account.username)
            try:
                response = self.session.get(url, timeout=timeout)
                response.raise_for_status()
                root = ET.fromstring(response.text)
            except Exception:
                continue
            for node in root.findall(".//item"):
                title = (node.findtext("title") or "").strip()
                description = (node.findtext("description") or "").strip()
                published_at = parse_pubdate(node.findtext("pubDate"))
                author_hint = self._extract_author_hint(title, description, account)
                if not self._author_matches_watchlist(author_hint, account):
                    continue
                if not self._within_lookback(published_at, now):
                    continue
                if published_at is not None and (now - published_at).total_seconds() > self.twitter_event_max_age_minutes * 60:
                    continue
                combined = " ".join(part for part in [title, description] if part)
                matched_symbols, matched_keywords = self._match_symbols_and_keywords(combined, account)
                if symbol not in matched_symbols:
                    continue
                base_score = score_text(combined)
                urgency = recency_weight(published_at, now, max(1, self.twitter_event_max_age_minutes / 60.0))
                credibility = author_weight(account.priority, account.reliability)
                relevance = relevance_score(combined, keywords)
                weight = urgency * credibility * relevance
                if abs(base_score * weight) < self.twitter_event_min_score:
                    continue
                event = TwitterEvent(
                    event_id=novelty_key(f"{account.username} {title}"),
                    source_platform="twitter",
                    author_username=account.username,
                    author_display_name=account.display_name,
                    author_category=account.category,
                    author_priority=account.priority,
                    author_reliability=account.reliability,
                    created_at=published_at,
                    fetched_at=now,
                    text=combined,
                    matched_symbols=matched_symbols,
                    matched_keywords=matched_keywords,
                    event_type=dominant_event_type([SentimentItem("twitter", "twitter", title, published_at, base_score, weight, relevance, event_tags(combined))]),
                    sentiment_score=base_score,
                    urgency_score=urgency,
                    credibility_score=credibility,
                    confirmation_state="unconfirmed",
                    raw_payload={"title": title, "description": description, "pubDate": node.findtext("pubDate") or ""},
                )
                items.append(
                    SentimentItem(
                        source="twitter",
                        source_name=account.display_name,
                        title=title or combined[:120],
                        published_at=published_at,
                        score=base_score,
                        weight=weight,
                        relevance=relevance,
                        event_tags=event_tags(combined),
                        author_username=account.username,
                        author_display_name=account.display_name,
                        author_category=account.category,
                        author_priority=account.priority,
                        author_reliability=account.reliability,
                        raw_payload=event.raw_payload,
                    )
                )
                if len(items) >= self.twitter_limit:
                    return items
        return items

    def _extract_author_hint(self, title: str, description: str, account: TwitterWatchAccount) -> str:
        for text in (title, description):
            match = re.search(r"@([A-Za-z0-9_]+)", text or "")
            if match:
                return normalize_username(match.group(1))
        return account.username

    def _author_matches_watchlist(self, author_hint: str, account: TwitterWatchAccount) -> bool:
        normalized = normalize_username(author_hint)
        if normalized in set(account.denylist):
            return False
        allowed = {account.username, *account.aliases}
        if normalized not in allowed:
            return normalized == account.username
        lowered_display = account.display_name.lower()
        parody_markers = ("parody", "fake", "fan", "impersonator")
        if any(marker in lowered_display for marker in parody_markers):
            return False
        return True

    def _match_symbols_and_keywords(self, text: str, account: TwitterWatchAccount) -> tuple[List[str], List[str]]:
        lowered = text.lower()
        matched_symbols: List[str] = []
        matched_keywords: List[str] = []
        for symbol, aliases in SYMBOL_KEYWORDS.items():
            extra = [f"${symbol.split('/', 1)[0].lower()}", f"#{symbol.split('/', 1)[0].lower()}"]
            if any(alias in lowered for alias in aliases + extra):
                matched_symbols.append(symbol)
                matched_keywords.extend([alias for alias in aliases if alias in lowered])
        for related in account.related_assets:
            if related not in matched_symbols:
                matched_symbols.append(related)
        if account.category in {"regulators", "politicians", "macro"} and "bitcoin" in lowered and "BTC/USD" not in matched_symbols:
            matched_symbols.append("BTC/USD")
        return matched_symbols, list(dict.fromkeys(matched_keywords))
