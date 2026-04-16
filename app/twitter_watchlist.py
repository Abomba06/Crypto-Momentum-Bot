import json
import pathlib
from typing import List

from app.event_models import TwitterWatchAccount


def load_watchlist(path: str) -> List[TwitterWatchAccount]:
    if not path:
        return []
    watchlist_path = pathlib.Path(path)
    if not watchlist_path.exists():
        return []
    try:
        payload = json.loads(watchlist_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    accounts = payload.get("accounts", payload if isinstance(payload, list) else [])
    results: List[TwitterWatchAccount] = []
    for item in accounts:
        username = str(item.get("username", "")).strip().lstrip("@").lower()
        if not username:
            continue
        results.append(
            TwitterWatchAccount(
                username=username,
                display_name=str(item.get("display_name", username)).strip(),
                category=str(item.get("category", "general")).strip().lower(),
                priority=float(item.get("priority", 1.0)),
                reliability=float(item.get("reliability", 1.0)),
                enabled=bool(item.get("enabled", True)),
                related_assets=[str(value).strip().upper() for value in item.get("related_assets", []) if str(value).strip()],
                tags=[str(value).strip().lower() for value in item.get("tags", []) if str(value).strip()],
                aliases=[str(value).strip().lstrip("@").lower() for value in item.get("aliases", []) if str(value).strip()],
                denylist=[str(value).strip().lstrip("@").lower() for value in item.get("denylist", []) if str(value).strip()],
            )
        )
    return [account for account in results if account.enabled]
