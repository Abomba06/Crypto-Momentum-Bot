from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class TwitterWatchAccount:
    username: str
    display_name: str
    category: str
    priority: float
    reliability: float
    enabled: bool = True
    related_assets: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    denylist: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class TwitterEvent:
    event_id: str
    source_platform: str
    author_username: str
    author_display_name: str
    author_category: str
    author_priority: float
    author_reliability: float
    created_at: Optional[datetime]
    fetched_at: datetime
    text: str
    matched_symbols: List[str]
    matched_keywords: List[str]
    event_type: str
    sentiment_score: float
    urgency_score: float
    credibility_score: float
    confirmation_state: str
    raw_payload: Dict[str, Any]
