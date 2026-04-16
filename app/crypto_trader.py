import csv
import json
import math
import os
import pathlib
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException
from urllib3.util.retry import Retry

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.models import Order
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest
from app.sentiment import SentimentClient, SentimentSnapshot, parse_keyword_map


load_dotenv()


SECTOR_MAP = {
    "BTC/USD": "majors",
    "ETH/USD": "majors",
    "SOL/USD": "layer1",
    "AVAX/USD": "layer1",
    "ADA/USD": "layer1",
    "MATIC/USD": "layer1",
    "DOGE/USD": "memes",
    "LINK/USD": "infrastructure",
    "UNI/USD": "defi",
    "LTC/USD": "payments",
    "XRP/USD": "payments",
}

SPREAD_BPS_MAP = {
    "BTC/USD": 4.0,
    "ETH/USD": 5.0,
    "SOL/USD": 8.0,
    "DOGE/USD": 10.0,
    "AVAX/USD": 10.0,
    "LINK/USD": 9.0,
    "ADA/USD": 9.0,
    "MATIC/USD": 10.0,
    "UNI/USD": 11.0,
    "LTC/USD": 8.0,
    "XRP/USD": 9.0,
}


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def today_str() -> str:
    return now_utc().date().isoformat()


def is_finite(value: Optional[float]) -> bool:
    return value is not None and math.isfinite(value)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


@dataclass(frozen=True)
class BotConfig:
    symbols: List[str]
    ltf_raw: str
    htf_raw: str
    donchian: int
    atr_len: int
    stop_atr: float
    tp1_atr: float
    tp2_atr: float
    trail_atr: float
    breakout_buffer_bps: float
    risk_pct: float
    per_coin_cap: float
    portfolio_cap: float
    trend_mode: str
    htf_fast: int
    htf_slow: int
    htf_slope_n: int
    ltf_fast_ema: int
    ltf_slow_ema: int
    rsi_len: int
    rsi_entry_min: float
    min_atr_pct: float
    volume_window: int
    min_volume_ratio: float
    warmup_ltf: int
    warmup_htf: int
    mute_secs: int
    cooldown_secs: int
    loop_secs: int
    logs_dir: pathlib.Path
    run_log: pathlib.Path
    trades_csv: pathlib.Path
    state_path: pathlib.Path
    initial_cash: float
    daily_dd_limit: float
    alpaca_data_url: str
    apca_key_id: str
    apca_secret_key: str
    trading_base_url: str
    default_exchange: str
    request_timeout_secs: int
    sentiment_enabled: bool
    sentiment_mode: str
    sentiment_sources: List[str]
    sentiment_lookback_hours: int
    sentiment_min_items: int
    sentiment_buy_threshold: float
    sentiment_sell_threshold: float
    sentiment_cache_secs: int
    sentiment_exit_on_bearish: bool
    sentiment_keyword_map: Dict[str, List[str]]
    sentiment_news_limit: int
    sentiment_twitter_limit: int
    sentiment_twitter_rss_url: str
    max_breakout_atr_extension: float
    max_entry_rsi: float
    ema_slope_lookback: int
    dashboard_symbols: int
    max_entries_per_loop: int
    max_portfolio_heat: float
    max_sector_exposure: float
    weekly_dd_limit: float
    max_consecutive_losses: int
    pullback_ema_buffer_atr: float
    shock_sentiment_threshold: float
    signals_csv: pathlib.Path
    research_report: pathlib.Path
    rs_lookback: int
    min_relative_strength: float
    compression_window: int
    compression_atr_ratio: float
    reversal_lookback: int
    min_execution_quality: float
    news_momentum_min_acceleration: float
    news_momentum_max_age_hours: int
    news_momentum_min_recent_items: int
    cross_asset_riskoff_penalty: float
    cross_asset_alt_strength_bonus: float

    @classmethod
    def from_env(cls) -> "BotConfig":
        symbols = [
            s.strip()
            for s in os.getenv(
                "CRYPTO_SYMBOLS",
                "BTC/USD,ETH/USD,SOL/USD,DOGE/USD,AVAX/USD,LINK/USD,ADA/USD,MATIC/USD,UNI/USD,LTC/USD,XRP/USD",
            ).split(",")
            if s.strip()
        ]
        donchian = int(os.getenv("DONCHIAN", "20"))
        atr_len = int(os.getenv("ATR_LEN", "14"))
        htf_fast = int(os.getenv("HTF_FAST", "50"))
        htf_slow = int(os.getenv("HTF_SLOW", "200"))
        htf_slope_n = int(os.getenv("HTF_SLOPE_LOOKBACK", "10"))
        ltf_fast_ema = int(os.getenv("LTF_FAST_EMA", "21"))
        ltf_slow_ema = int(os.getenv("LTF_SLOW_EMA", "55"))
        rsi_len = int(os.getenv("RSI_LEN", "14"))
        volume_window = int(os.getenv("VOLUME_WINDOW", "20"))
        logs_dir = pathlib.Path(os.getenv("LOG_DIR", "logs"))
        logs_dir.mkdir(exist_ok=True)

        cooldown_secs = int(os.getenv("COOLDOWN_SECS", os.getenv("COOLDOWN_MIN", "0")))
        if os.getenv("COOLDOWN_MIN"):
            cooldown_secs = int(float(os.getenv("COOLDOWN_MIN", "0")) * 60)

        state_path = pathlib.Path(os.getenv("STATE_PATH", str(pathlib.Path("state.json").resolve())))
        apca_key_id = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_KEY_ID", "")
        apca_secret_key = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY", "")
        if not apca_key_id or not apca_secret_key:
            raise RuntimeError(
                "Missing Alpaca API keys (APCA_API_KEY_ID/APCA_API_SECRET_KEY or ALPACA_KEY_ID/ALPACA_SECRET_KEY)"
            )

        warmup_ltf = int(
            os.getenv(
                "WARMUP_LTF",
                str(max(donchian + 5, atr_len + 5, ltf_slow_ema + 5, volume_window + 5, rsi_len + 5)),
            )
        )
        warmup_htf = int(os.getenv("WARMUP_HTF", str(max(htf_slow + htf_slope_n + 5, 220))))

        return cls(
            symbols=symbols,
            ltf_raw=os.getenv("CRYPTO_TIMEFRAME", "5Min").lower(),
            htf_raw=os.getenv("HTF_TIMEFRAME", "4Hour").lower(),
            donchian=donchian,
            atr_len=atr_len,
            stop_atr=float(os.getenv("STOP_ATR", "1.8")),
            tp1_atr=float(os.getenv("TP1_ATR", "1.5")),
            tp2_atr=float(os.getenv("TP2_ATR", "3.0")),
            trail_atr=float(os.getenv("TRAIL_ATR", "2.0")),
            breakout_buffer_bps=float(os.getenv("BREAKOUT_BUFFER_BPS", "8")),
            risk_pct=float(os.getenv("RISK_PCT", "0.01")),
            per_coin_cap=float(os.getenv("PER_COIN_CAP", "0.20")),
            portfolio_cap=float(os.getenv("PORTFOLIO_CAP", "0.50")),
            trend_mode=os.getenv("TREND_MODE", "strict").lower(),
            htf_fast=htf_fast,
            htf_slow=htf_slow,
            htf_slope_n=htf_slope_n,
            ltf_fast_ema=ltf_fast_ema,
            ltf_slow_ema=ltf_slow_ema,
            rsi_len=rsi_len,
            rsi_entry_min=float(os.getenv("RSI_ENTRY_MIN", "56")),
            min_atr_pct=float(os.getenv("MIN_ATR_PCT", "0.0035")),
            volume_window=volume_window,
            min_volume_ratio=float(os.getenv("MIN_VOLUME_RATIO", "1.10")),
            warmup_ltf=warmup_ltf,
            warmup_htf=warmup_htf,
            mute_secs=int(os.getenv("MUTE_SECS", "90")),
            cooldown_secs=cooldown_secs,
            loop_secs=int(os.getenv("LOOP_SECS", "15")),
            logs_dir=logs_dir,
            run_log=logs_dir / "run.log",
            trades_csv=logs_dir / "trades.csv",
            signals_csv=logs_dir / "signals.csv",
            research_report=logs_dir / "research_report.json",
            state_path=state_path,
            initial_cash=float(os.getenv("INITIAL_CASH", "100000")),
            daily_dd_limit=float(os.getenv("DAILY_DD_LIMIT", "1.0")),
            alpaca_data_url=os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets"),
            apca_key_id=apca_key_id,
            apca_secret_key=apca_secret_key,
            trading_base_url=os.getenv("APCA_API_BASE_URL", "").strip(),
            default_exchange=(os.getenv("ALPACA_CRYPTO_EXCHANGES") or "").strip(),
            request_timeout_secs=int(os.getenv("REQUEST_TIMEOUT_SECS", "20")),
            sentiment_enabled=os.getenv("SENTIMENT_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"},
            sentiment_mode=os.getenv("SENTIMENT_MODE", "confirm").strip().lower(),
            sentiment_sources=[
                source.strip().lower()
                for source in os.getenv("SENTIMENT_SOURCES", "news").split(",")
                if source.strip()
            ],
            sentiment_lookback_hours=int(os.getenv("SENTIMENT_LOOKBACK_HOURS", "24")),
            sentiment_min_items=int(os.getenv("SENTIMENT_MIN_ITEMS", "3")),
            sentiment_buy_threshold=float(os.getenv("SENTIMENT_BUY_THRESHOLD", "0.15")),
            sentiment_sell_threshold=float(os.getenv("SENTIMENT_SELL_THRESHOLD", "-0.15")),
            sentiment_cache_secs=int(os.getenv("SENTIMENT_CACHE_SECS", "300")),
            sentiment_exit_on_bearish=os.getenv("SENTIMENT_EXIT_ON_BEARISH", "true").strip().lower()
            in {"1", "true", "yes", "on"},
            sentiment_keyword_map=parse_keyword_map(os.getenv("SENTIMENT_KEYWORDS", "")),
            sentiment_news_limit=int(os.getenv("SENTIMENT_NEWS_LIMIT", "8")),
            sentiment_twitter_limit=int(os.getenv("SENTIMENT_TWITTER_LIMIT", "8")),
            sentiment_twitter_rss_url=os.getenv(
                "SENTIMENT_TWITTER_RSS_URL",
                "https://nitter.net/search/rss?f=tweets&q={query}",
            ).strip(),
            max_breakout_atr_extension=float(os.getenv("MAX_BREAKOUT_ATR_EXTENSION", "0.8")),
            max_entry_rsi=float(os.getenv("MAX_ENTRY_RSI", "72")),
            ema_slope_lookback=int(os.getenv("EMA_SLOPE_LOOKBACK", "3")),
            dashboard_symbols=int(os.getenv("DASHBOARD_SYMBOLS", "10")),
            max_entries_per_loop=int(os.getenv("MAX_ENTRIES_PER_LOOP", "2")),
            max_portfolio_heat=float(os.getenv("MAX_PORTFOLIO_HEAT", "0.06")),
            max_sector_exposure=float(os.getenv("MAX_SECTOR_EXPOSURE", "0.35")),
            weekly_dd_limit=float(os.getenv("WEEKLY_DD_LIMIT", "0.10")),
            max_consecutive_losses=int(os.getenv("MAX_CONSECUTIVE_LOSSES", "4")),
            pullback_ema_buffer_atr=float(os.getenv("PULLBACK_EMA_BUFFER_ATR", "0.35")),
            shock_sentiment_threshold=float(os.getenv("SHOCK_SENTIMENT_THRESHOLD", "-0.55")),
            rs_lookback=int(os.getenv("RS_LOOKBACK", "20")),
            min_relative_strength=float(os.getenv("MIN_RELATIVE_STRENGTH", "-0.02")),
            compression_window=int(os.getenv("COMPRESSION_WINDOW", "12")),
            compression_atr_ratio=float(os.getenv("COMPRESSION_ATR_RATIO", "0.85")),
            reversal_lookback=int(os.getenv("REVERSAL_LOOKBACK", "10")),
            min_execution_quality=float(os.getenv("MIN_EXECUTION_QUALITY", "0.55")),
            news_momentum_min_acceleration=float(os.getenv("NEWS_MOMENTUM_MIN_ACCELERATION", "0.12")),
            news_momentum_max_age_hours=int(os.getenv("NEWS_MOMENTUM_MAX_AGE_HOURS", "6")),
            news_momentum_min_recent_items=int(os.getenv("NEWS_MOMENTUM_MIN_RECENT_ITEMS", "2")),
            cross_asset_riskoff_penalty=float(os.getenv("CROSS_ASSET_RISKOFF_PENALTY", "0.72")),
            cross_asset_alt_strength_bonus=float(os.getenv("CROSS_ASSET_ALT_STRENGTH_BONUS", "0.08")),
        )


@dataclass(frozen=True)
class BrokerPosition:
    symbol: str
    qty: float
    qty_available: float
    avg_entry_price: float
    current_price: float
    market_value: float


@dataclass(frozen=True)
class AccountSnapshot:
    cash: float
    equity: float


@dataclass(frozen=True)
class EntrySignal:
    breakout_level: float
    stop: float
    tp1: float
    tp2: float
    trail_anchor: float
    atr_value: float
    volume_ratio: float
    rsi_value: float
    source: str
    confidence: float
    atr_pct: float
    ema_spread_pct: float
    reason: str


@dataclass(frozen=True)
class RegimeSnapshot:
    name: str
    risk_multiplier: float
    allow_trend: bool
    allow_pullback: bool
    allow_breakout: bool
    exit_aggression: float


@dataclass(frozen=True)
class CandidateSetup:
    symbol: str
    signal: EntrySignal
    regime: RegimeSnapshot
    sentiment: Optional[SentimentSnapshot]
    trend_score: float
    sector: str
    score: float
    last_price: float
    execution_quality: float
    relative_strength: float
    cross_asset_multiplier: float


@dataclass(frozen=True)
class CrossAssetContext:
    regime: str
    btc_trend_score: float
    eth_trend_score: float
    eth_vs_btc: float
    risk_multiplier: float
    majors_multiplier: float
    alts_multiplier: float


def default_state(initial_cash: float) -> Dict[str, Any]:
    return {
        "cash": initial_cash,
        "equity": initial_cash,
        "sym": {},
        "daily": {},
        "weekly": {},
        "meta": {"last_loop_at": None, "last_equity": initial_cash, "consecutive_losses": 0},
        "stats": {"closed": []},
    }


def log_line(config: BotConfig, message: str) -> None:
    with config.run_log.open("a", encoding="utf-8") as file_obj:
        file_obj.write(f"[{now_utc().isoformat()}] {message}\n")


def ensure_trades_csv(config: BotConfig) -> None:
    header = ["ts", "symbol", "action", "price_last", "avg_entry", "stop", "tp1", "tp2", "qty", "note"]
    is_new = not config.trades_csv.exists()
    with config.trades_csv.open("a", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        if is_new:
            writer.writerow(header)


def ensure_signals_csv(config: BotConfig) -> None:
    header = ["ts", "symbol", "setup", "regime", "score", "sentiment", "trend_score", "action", "reason"]
    is_new = not config.signals_csv.exists()
    with config.signals_csv.open("a", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        if is_new:
            writer.writerow(header)


def write_trade(
    config: BotConfig,
    symbol: str,
    action: str,
    price_last: float,
    avg_entry: float,
    stop: float,
    tp1: float,
    tp2: float,
    qty: float,
    note: str = "",
) -> None:
    ensure_trades_csv(config)
    with config.trades_csv.open("a", newline="", encoding="utf-8") as file_obj:
        csv.writer(file_obj).writerow(
            [
                now_utc().isoformat(),
                symbol,
                action,
                f"{safe_float(price_last):.6f}",
                f"{safe_float(avg_entry):.6f}",
                f"{safe_float(stop):.6f}",
                f"{safe_float(tp1):.6f}",
                f"{safe_float(tp2):.6f}",
                f"{safe_float(qty):.8f}",
                note,
            ]
        )


def write_signal_journal(
    config: BotConfig,
    symbol: str,
    setup: str,
    regime: str,
    score: float,
    sentiment_score: Optional[float],
    trend_value: float,
    action: str,
    reason: str,
) -> None:
    ensure_signals_csv(config)
    with config.signals_csv.open("a", newline="", encoding="utf-8") as file_obj:
        csv.writer(file_obj).writerow(
            [
                now_utc().isoformat(),
                symbol,
                setup,
                regime,
                f"{safe_float(score):.4f}",
                "" if sentiment_score is None else f"{safe_float(sentiment_score):.4f}",
                f"{safe_float(trend_value):.4f}",
                action,
                reason,
            ]
        )


def print_dashboard(
    config: BotConfig,
    state: Dict[str, Any],
    positions: Dict[str, BrokerPosition],
    open_order_symbols: Set[str],
) -> None:
    timestamp = now_utc().strftime("%Y-%m-%d %H:%M:%S UTC")
    daily = state.get("daily", {})
    start_equity = safe_float(daily.get("start_equity"), state["equity"])
    day_return = ((state["equity"] / max(start_equity, 1e-9)) - 1.0) * 100.0
    lines = [
        "",
        f"[{timestamp}] Crypto bot running",
        f"Equity: ${state['equity']:.2f} | Cash: ${state['cash']:.2f} | Day PnL: {day_return:+.2f}% | Universe: {len(config.symbols)}",
        f"Daily halt: {'ON' if daily.get('halt', False) else 'off'} | Weekly halt: {'ON' if state.get('weekly', {}).get('halt', False) else 'off'} | Open order symbols: {', '.join(sorted(open_order_symbols)) if open_order_symbols else 'none'}",
        f"Portfolio heat: {safe_float(state.get('meta', {}).get('portfolio_heat')):.2%} | Loss streak: {int(state.get('meta', {}).get('consecutive_losses', 0))} | Candidates: {int(state.get('meta', {}).get('candidate_count', 0))}",
    ]
    cross_asset = state.get("meta", {}).get("cross_asset", {})
    if cross_asset:
        lines.append(
            f"Cross-asset: {cross_asset.get('regime', 'neutral')} | BTC trend {safe_float(cross_asset.get('btc_trend_score')):+.2f} "
            f"| ETH trend {safe_float(cross_asset.get('eth_trend_score')):+.2f} | ETH/BTC RS {safe_float(cross_asset.get('eth_vs_btc')):+.3f}"
        )
    reconcile = state.get("meta", {}).get("reconciliation", {})
    if reconcile:
        lines.append(
            f"Reconcile: restored={int(reconcile.get('restored', 0))} "
            f"cleared={int(reconcile.get('cleared', 0))} adjusted={int(reconcile.get('mismatched_qty', 0))}"
        )
    top_candidates = state.get("meta", {}).get("top_candidates", [])
    if top_candidates:
        top_line = ", ".join(
            f"{item.get('symbol')} {item.get('setup')} {safe_float(item.get('score')):.2f}"
            for item in top_candidates[:3]
        )
        lines.append(f"Top candidates: {top_line}")
    alerts = state.get("meta", {}).get("alerts", [])
    if alerts:
        lines.append(f"Alerts: {' | '.join(str(item) for item in alerts[:3])}")

    active_positions = [position for position in positions.values() if position.qty > 0]
    if active_positions:
        lines.append("Open positions:")
        for position in sorted(active_positions, key=lambda item: item.symbol):
            lines.append(
                f"  {position.symbol}: qty={position.qty:.8f} avg={position.avg_entry_price:.6f} "
                f"last={position.current_price:.6f} value=${position.market_value:.2f}"
            )
    else:
        lines.append("Open positions: none")

    snapshots = []
    for symbol in config.symbols:
        snapshot = state.get("sym", {}).get(symbol, {}).get("snapshot")
        if snapshot:
            snapshots.append(snapshot)

    if snapshots:
        lines.append("Symbol monitor:")
        header = "  Symbol    Last      Pos$    UPNL%  Regime   XAst Trend  RS     Sentiment          Setup         Action"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        ranked = sorted(
            snapshots,
            key=lambda item: (
                item.get("has_position", False),
                item.get("entry_ready", False),
                safe_float(item.get("sentiment_score"), -9.0),
                safe_float(item.get("trend_score"), -9.0),
            ),
            reverse=True,
        )
        for snapshot in ranked[: max(1, config.dashboard_symbols)]:
            sentiment_text = snapshot.get("sentiment_label", "n/a")
            sentiment_score = snapshot.get("sentiment_score")
            if sentiment_score is not None and math.isfinite(float(sentiment_score)):
                sentiment_text = f"{sentiment_text} {float(sentiment_score):+.2f}"
            unrealized = snapshot.get("unrealized_pct")
            pnl_text = "--"
            if unrealized is not None and math.isfinite(float(unrealized)):
                pnl_text = f"{float(unrealized):+5.2f}"
            lines.append(
                f"  {snapshot.get('symbol', ''):<8} {safe_float(snapshot.get('last_price')):>8.3f} "
                f"{safe_float(snapshot.get('position_value')):>7.0f} {pnl_text:>7} "
                f"{snapshot.get('regime', 'n/a'):<8} {snapshot.get('cross_asset_regime', 'n/a'):<4} {snapshot.get('trend_label', '?'):<6} "
                f"{safe_float(snapshot.get('relative_strength')):+5.2f} {sentiment_text:<18} "
                f"{snapshot.get('setup_label', 'idle'):<12} {snapshot.get('last_action', 'watch')}"
            )
            headlines = snapshot.get("top_headlines", [])
            if headlines:
                lines.append(f"    headlines: {headlines[0][:120]}")
            reason = snapshot.get("reason")
            if reason:
                lines.append(f"    reason: {reason}")

    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


def build_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.75,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class AlpacaCryptoDataClient:
    _TF_MAP = {
        "1min": "1Min",
        "3min": "3Min",
        "5min": "5Min",
        "15min": "15Min",
        "30min": "30Min",
        "1hour": "1Hour",
        "1h": "1Hour",
        "2hour": "2Hour",
        "2h": "2Hour",
        "4hour": "4Hour",
        "4h": "4Hour",
        "6hour": "6Hour",
        "6h": "6Hour",
        "12hour": "12Hour",
        "12h": "12Hour",
        "1day": "1Day",
        "day": "1Day",
        "1d": "1Day",
    }

    def __init__(self, config: BotConfig):
        self.config = config
        self.session = build_session()
        self.headers = {
            "APCA-API-KEY-ID": config.apca_key_id,
            "APCA-API-SECRET-KEY": config.apca_secret_key,
        }

    def map_timeframe(self, timeframe: str) -> str:
        key = timeframe.lower()
        if key not in self._TF_MAP:
            raise ValueError(f"Unsupported timeframe for Alpaca: {timeframe}")
        return self._TF_MAP[key]

    def _tf_minutes(self, timeframe: str) -> int:
        if timeframe.endswith("Min"):
            return int(timeframe.replace("Min", ""))
        if timeframe.endswith("Hour"):
            return int(timeframe.replace("Hour", "")) * 60
        if timeframe.endswith("Day"):
            return int(timeframe.replace("Day", "")) * 60 * 24
        return 15

    def _iso(self, dt: datetime) -> str:
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _get_json(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        response = self.session.get(
            self.config.alpaca_data_url + path,
            params=params,
            headers=self.headers,
            timeout=self.config.request_timeout_secs,
        )
        response.raise_for_status()
        return response.json()

    def _drop_last_if_open_bars(self, bars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return bars[:-1] if bars else bars

    def _try_bars(self, symbol: str, timeframe: str, limit: int) -> List[Dict[str, Any]]:
        mins = self._tf_minutes(timeframe)
        bars_needed = max(limit + 50, 600)
        from_when = self._iso(now_utc() - timedelta(minutes=bars_needed * mins))

        params_list = []
        if self.config.default_exchange:
            params_list.append(
                {
                    "symbols": f"{symbol}:{self.config.default_exchange}",
                    "timeframe": timeframe,
                    "start": from_when,
                    "limit": min(10000, bars_needed + 200),
                }
            )
        params_list.append(
            {
                "symbols": symbol,
                "timeframe": timeframe,
                "start": from_when,
                "limit": min(10000, bars_needed + 200),
            }
        )

        last_error: Optional[Exception] = None
        for params in params_list:
            try:
                payload = self._get_json("/v1beta3/crypto/us/bars", params)
            except RequestException as exc:
                last_error = exc
                continue

            bars_map = payload.get("bars", {})
            bars = (
                bars_map.get(symbol)
                or (bars_map.get(f"{symbol}:{self.config.default_exchange}") if self.config.default_exchange else None)
                or (bars_map.get(next(iter(bars_map))) if bars_map else None)
            )
            if bars:
                return bars

        if last_error:
            raise last_error
        return []

    def probe_symbol(self, symbol: str) -> bool:
        try:
            bars = self._try_bars(symbol, self.map_timeframe(self.config.htf_raw), 10)
        except Exception as exc:
            log_line(self.config, f"[WARN] {symbol} probe failed: {type(exc).__name__}: {exc}")
            return False
        if bars:
            return True
        log_line(self.config, f"[WARN] {symbol} has 0 bars from Alpaca; dropping from universe.")
        return False

    def fetch_htf_closes(self, symbol: str, limit: int) -> List[float]:
        bars = self._drop_last_if_open_bars(self._try_bars(symbol, self.map_timeframe(self.config.htf_raw), limit))
        return [float(bar["c"]) for bar in bars]

    def fetch_ltf_ohlcv(self, symbol: str, limit: int) -> Tuple[List[float], List[float], List[float], List[float]]:
        bars = self._drop_last_if_open_bars(self._try_bars(symbol, self.map_timeframe(self.config.ltf_raw), limit))
        highs = [float(bar["h"]) for bar in bars]
        lows = [float(bar["l"]) for bar in bars]
        closes = [float(bar["c"]) for bar in bars]
        volumes = [float(bar.get("v", 0.0)) for bar in bars]
        return highs, lows, closes, volumes

    def fetch_last_price(self, symbol: str) -> float:
        try:
            if self.config.default_exchange:
                payload = self._get_json(
                    "/v1beta3/crypto/us/trades/latest",
                    {"symbols": f"{symbol}:{self.config.default_exchange}"},
                )
                trade = payload.get("trades", {}).get(f"{symbol}:{self.config.default_exchange}")
                if trade and "p" in trade:
                    return float(trade["p"])

            payload = self._get_json("/v1beta3/crypto/us/trades/latest", {"symbols": symbol})
            trade = payload.get("trades", {}).get(symbol)
            if trade and "p" in trade:
                return float(trade["p"])
        except RequestException:
            return float("nan")
        return float("nan")


class AlpacaPaperBroker:
    def __init__(self, config: BotConfig):
        self.config = config
        self.trading = TradingClient(
            config.apca_key_id,
            config.apca_secret_key,
            paper=True,
            url_override=config.trading_base_url or None,
        )

    def get_account_snapshot(self) -> AccountSnapshot:
        account = self.trading.get_account()
        return AccountSnapshot(
            cash=safe_float(account.cash),
            equity=safe_float(account.equity or account.portfolio_value),
        )

    def get_positions(self, symbols: List[str]) -> Dict[str, BrokerPosition]:
        positions: Dict[str, BrokerPosition] = {}
        symbol_set = set(symbols)
        for position in self.trading.get_all_positions():
            if position.symbol not in symbol_set:
                continue
            qty = safe_float(position.qty)
            qty_available = safe_float(position.qty_available, qty)
            positions[position.symbol] = BrokerPosition(
                symbol=position.symbol,
                qty=qty,
                qty_available=qty_available,
                avg_entry_price=safe_float(position.avg_entry_price),
                current_price=safe_float(position.current_price),
                market_value=safe_float(position.market_value, qty * safe_float(position.current_price)),
            )
        return positions

    def get_open_order_symbols(self, symbols: List[str]) -> Set[str]:
        try:
            orders = self.trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=symbols))
        except Exception:
            return set()
        return {order.symbol for order in orders if order.symbol}

    def submit_market_buy(self, symbol: str, qty: float) -> Order:
        order = MarketOrderRequest(
            symbol=symbol,
            qty=round(qty, 8),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
        )
        return self.trading.submit_order(order)

    def submit_market_sell(self, symbol: str, qty: float) -> Order:
        order = MarketOrderRequest(
            symbol=symbol,
            qty=round(qty, 8),
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
        )
        return self.trading.submit_order(order)


def ema(seq: List[float], n: int) -> float:
    if len(seq) < n:
        return float("nan")
    alpha = 2.0 / (n + 1.0)
    value = seq[0]
    for price in seq[1:]:
        value = alpha * price + (1.0 - alpha) * value
    return value


def ema_series(seq: List[float], n: int) -> List[float]:
    if len(seq) < n:
        return []
    alpha = 2.0 / (n + 1.0)
    value = seq[0]
    values = [value]
    for price in seq[1:]:
        value = alpha * price + (1.0 - alpha) * value
        values.append(value)
    return values


def rsi(seq: List[float], n: int) -> float:
    if len(seq) < n + 1:
        return float("nan")
    gains = []
    losses = []
    for idx in range(1, len(seq)):
        delta = seq[idx] - seq[idx - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
    avg_gain = sum(gains[:n]) / n
    avg_loss = sum(losses[:n]) / n
    for gain, loss in zip(gains[n:], losses[n:]):
        avg_gain = ((avg_gain * (n - 1)) + gain) / n
        avg_loss = ((avg_loss * (n - 1)) + loss) / n
    rs = avg_gain / max(avg_loss, 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def sma(seq: List[float], n: int) -> float:
    if len(seq) < n:
        return float("nan")
    return sum(seq[-n:]) / float(n)


def donchian(seq: List[float], n: int) -> Tuple[float, float]:
    if len(seq) < n:
        return float("nan"), float("nan")
    window = seq[-n:]
    return max(window), min(window)


def true_ranges(highs: List[float], lows: List[float], closes: List[float]) -> List[float]:
    values = []
    for idx in range(1, len(closes)):
        hi, lo, prev_close = highs[idx], lows[idx], closes[idx - 1]
        values.append(max(hi - lo, abs(hi - prev_close), abs(lo - prev_close)))
    return values


def atr(highs: List[float], lows: List[float], closes: List[float], n: int) -> float:
    if len(closes) < n + 1:
        return float("nan")
    ranges = true_ranges(highs, lows, closes)
    if len(ranges) < n:
        return float("nan")
    return sum(ranges[-n:]) / float(n)


def crossed_up(prev: Optional[float], now: Optional[float], level: Optional[float]) -> bool:
    return is_finite(prev) and is_finite(now) and is_finite(level) and prev < level <= now


def crossed_down(prev: Optional[float], now: Optional[float], level: Optional[float]) -> bool:
    return is_finite(prev) and is_finite(now) and is_finite(level) and prev > level >= now


def load_state(config: BotConfig) -> Dict[str, Any]:
    if config.state_path.exists():
        try:
            state = json.loads(config.state_path.read_text(encoding="utf-8"))
            state.setdefault("cash", config.initial_cash)
            state.setdefault("equity", state.get("cash", config.initial_cash))
            state.setdefault("sym", {})
            state.setdefault("daily", {})
            state.setdefault("weekly", {})
            state.setdefault("meta", {"last_loop_at": None, "last_equity": state.get("equity", config.initial_cash), "consecutive_losses": 0})
            state.setdefault("stats", {"closed": []})
            return state
        except Exception as exc:
            log_line(config, f"STATE_LOAD_ERROR: {type(exc).__name__}: {exc}")
    return deepcopy(default_state(config.initial_cash))


def save_state(config: BotConfig, state: Dict[str, Any]) -> None:
    config.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def get_sym_state(state: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    sym_state = state["sym"].get(symbol)
    if sym_state is None:
        sym_state = {
            "prev_price": None,
            "last_fire_ts": None,
            "tp1_hit": False,
            "warm_ltf_ok": False,
            "warm_htf_ok": False,
            "high_water": None,
            "last_entry_price": None,
        }
        state["sym"][symbol] = sym_state
    return sym_state


def reset_trade_state(sym_state: Dict[str, Any]) -> None:
    sym_state["tp1_hit"] = False
    sym_state["high_water"] = None
    sym_state["last_entry_price"] = None
    sym_state["live_stop"] = None
    sym_state["entry_source"] = None
    sym_state["entry_regime"] = None


def can_fire(config: BotConfig, sym_state: Dict[str, Any]) -> bool:
    last = sym_state.get("last_fire_ts")
    if last is None:
        return True
    return (now_utc().timestamp() - float(last)) >= config.cooldown_secs


def set_fired(sym_state: Dict[str, Any]) -> None:
    sym_state["last_fire_ts"] = now_utc().timestamp()


def total_notional(positions: Dict[str, BrokerPosition], prices: Dict[str, float]) -> float:
    total = 0.0
    for symbol, position in positions.items():
        price = prices.get(symbol)
        if is_finite(price):
            total += abs(position.qty * float(price))
        else:
            total += abs(position.market_value)
    return total


def init_daily_controls(config: BotConfig, state: Dict[str, Any]) -> None:
    daily = state.setdefault("daily", {})
    if daily.get("date") != today_str():
        daily["date"] = today_str()
        daily["start_equity"] = state.get("equity", config.initial_cash)
        daily["halt"] = False


def week_str() -> str:
    year, week, _ = now_utc().isocalendar()
    return f"{year}-W{week:02d}"


def init_weekly_controls(config: BotConfig, state: Dict[str, Any]) -> None:
    weekly = state.setdefault("weekly", {})
    if weekly.get("week") != week_str():
        weekly["week"] = week_str()
        weekly["start_equity"] = state.get("equity", config.initial_cash)
        weekly["halt"] = False


def update_loss_streak(state: Dict[str, Any], new_equity: float) -> None:
    meta = state.setdefault("meta", {})
    last_equity = safe_float(meta.get("last_equity"), new_equity)
    if new_equity < last_equity - 1e-9:
        meta["consecutive_losses"] = int(meta.get("consecutive_losses", 0)) + 1
    elif new_equity > last_equity + 1e-9:
        meta["consecutive_losses"] = 0
    meta["last_equity"] = new_equity


def size_for_risk(config: BotConfig, cash: float, last: float, stop: float, equity: float, allocated_now: float) -> float:
    if not is_finite(last) or not is_finite(stop):
        return 0.0
    risk_dollars = equity * config.risk_pct
    risk_per_unit = max(1e-9, last - stop)
    qty_risk_based = risk_dollars / risk_per_unit

    if config.per_coin_cap >= 1.0:
        max_qty_by_coin_cap = config.per_coin_cap / max(1e-9, last)
    else:
        max_qty_by_coin_cap = (cash * config.per_coin_cap) / max(1e-9, last)

    max_port_notional = equity * config.portfolio_cap
    room_left = max(0.0, max_port_notional - allocated_now)
    max_qty_by_port = room_left / max(1e-9, last)
    max_qty_by_cash = cash / max(1e-9, last)

    return max(0.0, min(qty_risk_based, max_qty_by_coin_cap, max_qty_by_port, max_qty_by_cash))


def trend_score(fast: float, slow: float, slope: float, last_close: float) -> float:
    if not (is_finite(fast) and is_finite(slow) and is_finite(slope) and is_finite(last_close)):
        return 0.0
    score = 0.0
    score += (fast - slow) / max(abs(slow), 1e-9)
    score += slope / max(abs(last_close), 1e-9)
    return score * 100.0


def summarize_trend(score: float, trend_ok: bool) -> str:
    if trend_ok and score >= 1.0:
        return "strong"
    if trend_ok:
        return "up"
    if score <= -1.0:
        return "down"
    return "mixed"


def compute_position_value(position: Optional[BrokerPosition], last: float) -> float:
    if position is None or position.qty <= 0:
        return 0.0
    return position.qty * last if is_finite(last) else position.market_value


def compute_unrealized_pct(position: Optional[BrokerPosition], last: float) -> Optional[float]:
    if position is None or position.qty <= 0 or not is_finite(position.avg_entry_price) or position.avg_entry_price <= 0:
        return None
    return ((last / position.avg_entry_price) - 1.0) * 100.0


def sector_for_symbol(symbol: str) -> str:
    return SECTOR_MAP.get(symbol, "other")


def default_cross_asset_context() -> CrossAssetContext:
    return CrossAssetContext(
        regime="neutral",
        btc_trend_score=0.0,
        eth_trend_score=0.0,
        eth_vs_btc=0.0,
        risk_multiplier=1.0,
        majors_multiplier=1.0,
        alts_multiplier=1.0,
    )


def compute_cross_asset_context(config: BotConfig, benchmark_context: Dict[str, List[float]]) -> CrossAssetContext:
    btc_closes = benchmark_context.get("BTC/USD")
    eth_closes = benchmark_context.get("ETH/USD")
    if not btc_closes and not eth_closes:
        return default_cross_asset_context()

    btc_trend_score = 0.0
    eth_trend_score = 0.0
    btc_trend_ok = False
    eth_trend_ok = False
    if btc_closes and len(btc_closes) >= max(config.htf_slow + config.htf_slope_n, config.warmup_htf):
        btc_trend_ok, btc_fast, btc_slow, btc_slope = compute_trend_ok(config, btc_closes)
        btc_trend_score = trend_score(btc_fast, btc_slow, btc_slope, btc_closes[-1])
    if eth_closes and len(eth_closes) >= max(config.htf_slow + config.htf_slope_n, config.warmup_htf):
        eth_trend_ok, eth_fast, eth_slow, eth_slope = compute_trend_ok(config, eth_closes)
        eth_trend_score = trend_score(eth_fast, eth_slow, eth_slope, eth_closes[-1])

    eth_vs_btc = relative_strength_score(eth_closes or [], btc_closes, config.rs_lookback)
    if btc_trend_ok and eth_trend_ok and eth_vs_btc > 0.02:
        return CrossAssetContext("risk-on", btc_trend_score, eth_trend_score, eth_vs_btc, 1.06, 1.02, 1.08)
    if btc_trend_ok and btc_trend_score >= 1.0 and eth_vs_btc <= -0.01:
        return CrossAssetContext("majors-lead", btc_trend_score, eth_trend_score, eth_vs_btc, 0.98, 1.04, 0.92)
    if btc_trend_score <= -0.6 and eth_trend_score <= -0.4:
        penalty = max(0.45, min(0.95, config.cross_asset_riskoff_penalty))
        return CrossAssetContext("risk-off", btc_trend_score, eth_trend_score, eth_vs_btc, penalty, 0.88, penalty)
    if eth_trend_ok and eth_vs_btc > 0.04:
        alt_bonus = max(0.0, min(0.20, config.cross_asset_alt_strength_bonus))
        return CrossAssetContext("alts-lead", btc_trend_score, eth_trend_score, eth_vs_btc, 1.02, 0.98, 1.02 + alt_bonus)
    return CrossAssetContext("neutral", btc_trend_score, eth_trend_score, eth_vs_btc, 1.0, 1.0, 1.0)


def cross_asset_multiplier(symbol: str, cross_asset: CrossAssetContext) -> float:
    if sector_for_symbol(symbol) == "majors":
        return cross_asset.majors_multiplier
    return cross_asset.alts_multiplier


def portfolio_heat(positions: Dict[str, BrokerPosition], state: Dict[str, Any], prices_now: Dict[str, float], equity: float) -> float:
    if equity <= 0:
        return 0.0
    total_risk = 0.0
    for symbol, position in positions.items():
        if position.qty <= 0:
            continue
        last = prices_now.get(symbol, position.current_price)
        entry = position.avg_entry_price
        sym_state = state.get("sym", {}).get(symbol, {})
        stored_stop = safe_float(sym_state.get("live_stop"), entry * 0.95)
        risk_per_unit = max(0.0, entry - stored_stop)
        total_risk += risk_per_unit * position.qty
        if not is_finite(last):
            total_risk += 0.0
    return total_risk / equity


def sector_exposure(positions: Dict[str, BrokerPosition], prices_now: Dict[str, float], equity: float) -> Dict[str, float]:
    exposures: Dict[str, float] = {}
    if equity <= 0:
        return exposures
    for symbol, position in positions.items():
        if position.qty <= 0:
            continue
        last = prices_now.get(symbol, position.current_price)
        notional = abs(position.qty * last) if is_finite(last) else abs(position.market_value)
        sector = sector_for_symbol(symbol)
        exposures[sector] = exposures.get(sector, 0.0) + (notional / equity)
    return exposures


def detect_regime(
    trend_ok: bool,
    trend_value: float,
    atr_pct: float,
    vol_ratio: float,
    sentiment_snapshot: Optional[SentimentSnapshot],
) -> RegimeSnapshot:
    sentiment_score = 0.0 if sentiment_snapshot is None else sentiment_snapshot.score
    acceleration = 0.0 if sentiment_snapshot is None else sentiment_snapshot.acceleration
    if sentiment_score <= -0.55:
        return RegimeSnapshot("panic", 0.35, False, False, False, 1.35)
    if atr_pct >= 0.05 and sentiment_score >= 0.25:
        return RegimeSnapshot("euphoria", 0.55, True, False, False, 1.25)
    if trend_ok and trend_value >= 1.0 and atr_pct >= 0.01:
        return RegimeSnapshot("trend", 1.00, True, True, True, 1.00)
    if atr_pct < 0.007 and (not math.isfinite(vol_ratio) or vol_ratio < 1.0):
        return RegimeSnapshot("chop", 0.0, False, False, False, 0.85)
    if trend_ok and acceleration > 0.1:
        return RegimeSnapshot("high-vol breakout", 0.85, True, False, True, 1.10)
    if trend_ok:
        return RegimeSnapshot("uptrend-lite", 0.80, True, True, False, 0.95)
    return RegimeSnapshot("risk-off", 0.25, False, False, False, 1.15)


def update_symbol_snapshot(
    state: Dict[str, Any],
    symbol: str,
    *,
    last: float,
    position: Optional[BrokerPosition],
    trend_ok: bool,
    trend_value: float,
    sentiment_snapshot: Optional[SentimentSnapshot],
    entry_signal: Optional[EntrySignal],
    last_action: str,
    regime: Optional[RegimeSnapshot] = None,
    relative_strength: float = 0.0,
    execution_quality: float = 1.0,
    cross_asset: Optional[CrossAssetContext] = None,
) -> None:
    sym_state = get_sym_state(state, symbol)
    sym_state["snapshot"] = {
        "symbol": symbol,
        "last_price": last,
        "has_position": bool(position and position.qty > 0),
        "position_value": compute_position_value(position, last),
        "unrealized_pct": compute_unrealized_pct(position, last),
        "trend_ok": trend_ok,
        "trend_score": trend_value,
        "trend_label": summarize_trend(trend_value, trend_ok),
        "regime": None if regime is None else regime.name,
        "relative_strength": relative_strength,
        "execution_quality": execution_quality,
        "cross_asset_regime": None if cross_asset is None else cross_asset.regime,
        "cross_asset_multiplier": 1.0 if cross_asset is None else cross_asset_multiplier(symbol, cross_asset),
        "sentiment_score": None if sentiment_snapshot is None else sentiment_snapshot.score,
        "sentiment_label": "n/a" if sentiment_snapshot is None else sentiment_snapshot.label,
        "sentiment_samples": 0 if sentiment_snapshot is None else sentiment_snapshot.sample_size,
        "top_headlines": [] if sentiment_snapshot is None else sentiment_snapshot.top_headlines[:2],
        "event_counts": {} if sentiment_snapshot is None else sentiment_snapshot.event_counts,
        "entry_ready": entry_signal is not None,
        "setup_label": "idle" if entry_signal is None else f"{entry_signal.source} {entry_signal.confidence:.2f}",
        "reason": "" if entry_signal is None else entry_signal.reason,
        "candidate_score": None,
        "last_action": last_action,
        "updated_at": now_utc().isoformat(),
    }


def set_snapshot_candidate_score(state: Dict[str, Any], symbol: str, candidate_score: Optional[float]) -> None:
    snapshot = state.get("sym", {}).get(symbol, {}).get("snapshot")
    if snapshot is not None:
        snapshot["candidate_score"] = candidate_score


def update_trade_stats(state: Dict[str, Any], symbol: str, source: str, regime: str, pnl_value: float, pnl_pct: float) -> None:
    stats = state.setdefault("stats", {"closed": []})
    record = {
        "ts": now_utc().isoformat(),
        "symbol": symbol,
        "source": source,
        "regime": regime,
        "pnl_value": pnl_value,
        "pnl_pct": pnl_pct,
    }
    closed = stats.setdefault("closed", [])
    closed.append(record)
    if len(closed) > 500:
        del closed[:-500]


def summarize_bucket(records: List[Dict[str, Any]]) -> Dict[str, float]:
    count = len(records)
    wins = [item for item in records if safe_float(item.get("pnl_value")) > 0]
    losses = [item for item in records if safe_float(item.get("pnl_value")) < 0]
    gross_wins = sum(safe_float(item.get("pnl_value")) for item in wins)
    gross_losses = sum(safe_float(item.get("pnl_value")) for item in losses)
    avg_pnl = sum(safe_float(item.get("pnl_value")) for item in records) / max(count, 1)
    avg_win = gross_wins / max(len(wins), 1)
    avg_loss = sum(safe_float(item.get("pnl_value")) for item in losses) / max(len(losses), 1)
    expectancy = (len(wins) / max(count, 1)) * avg_win + (len(losses) / max(count, 1)) * avg_loss
    return {
        "count": count,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / max(count, 1),
        "avg_pnl": avg_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "gross_wins": gross_wins,
        "gross_losses": gross_losses,
    }


def build_parameter_health(
    summary: Dict[str, float],
    by_source: Dict[str, Dict[str, float]],
    by_regime: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    flags: List[str] = []
    weak_sources = [
        source
        for source, bucket in by_source.items()
        if int(bucket.get("count", 0)) >= 3 and safe_float(bucket.get("expectancy")) < 0
    ]
    weak_regimes = [
        regime
        for regime, bucket in by_regime.items()
        if int(bucket.get("count", 0)) >= 3 and safe_float(bucket.get("expectancy")) < 0
    ]

    if int(summary.get("count", 0)) >= 6 and safe_float(summary.get("expectancy")) < 0:
        flags.append("negative_expectancy")
    if safe_float(summary.get("win_rate")) < 0.35 and int(summary.get("count", 0)) >= 6:
        flags.append("low_win_rate")
    if weak_sources:
        flags.append("weak_sources")
    if weak_regimes:
        flags.append("weak_regimes")

    status = "healthy"
    if "negative_expectancy" in flags or "low_win_rate" in flags:
        status = "degraded"
    elif flags:
        status = "watch"
    return {
        "status": status,
        "flags": flags,
        "weak_sources": weak_sources,
        "weak_regimes": weak_regimes,
    }


def build_runtime_alerts(config: BotConfig, state: Dict[str, Any], report: Dict[str, Any]) -> List[str]:
    alerts: List[str] = []
    daily = state.get("daily", {})
    weekly = state.get("weekly", {})
    meta = state.get("meta", {})
    if daily.get("halt", False):
        alerts.append("Daily drawdown halt is active.")
    if weekly.get("halt", False):
        alerts.append("Weekly drawdown halt is active.")
    if int(meta.get("consecutive_losses", 0)) >= max(1, config.max_consecutive_losses - 1):
        alerts.append(f"Loss streak is elevated at {int(meta.get('consecutive_losses', 0))}.")
    reconcile = meta.get("reconciliation", {})
    if sum(int(reconcile.get(key, 0)) for key in ("restored", "cleared", "mismatched_qty")) > 0:
        alerts.append("Broker reconciliation adjusted state this loop.")
    if int(meta.get("no_candidate_loops", 0)) >= 3:
        alerts.append(f"No ranked candidates for {int(meta.get('no_candidate_loops', 0))} loops.")
    cross_asset = meta.get("cross_asset", {})
    if cross_asset.get("regime") == "risk-off":
        alerts.append("Cross-asset context is risk-off.")

    health = report.get("health", {})
    if health.get("status") == "degraded":
        alerts.append("Research expectancy is degraded.")
    elif health.get("status") == "watch":
        alerts.append("Research health is on watch.")
    return alerts[:6]


def build_research_report(state: Dict[str, Any]) -> Dict[str, Any]:
    closed = state.get("stats", {}).get("closed", [])
    summary = summarize_bucket(closed)
    by_source_records: Dict[str, List[Dict[str, Any]]] = {}
    by_regime_records: Dict[str, List[Dict[str, Any]]] = {}
    by_symbol_records: Dict[str, List[Dict[str, Any]]] = {}
    by_hour_records: Dict[str, List[Dict[str, Any]]] = {}
    for item in closed:
        by_source_records.setdefault(item.get("source", "unknown"), []).append(item)
        by_regime_records.setdefault(item.get("regime", "unknown"), []).append(item)
        by_symbol_records.setdefault(item.get("symbol", "unknown"), []).append(item)
        hour_key = "unknown"
        ts = item.get("ts")
        try:
            if ts:
                hour_key = f"{datetime.fromisoformat(ts).hour:02d}"
        except ValueError:
            hour_key = "unknown"
        by_hour_records.setdefault(hour_key, []).append(item)
    by_source = {key: summarize_bucket(records) for key, records in by_source_records.items()}
    by_regime = {key: summarize_bucket(records) for key, records in by_regime_records.items()}
    by_symbol = {key: summarize_bucket(records) for key, records in by_symbol_records.items()}
    by_hour = {key: summarize_bucket(records) for key, records in by_hour_records.items()}
    health = build_parameter_health(summary, by_source, by_regime)
    return {
        "generated_at": now_utc().isoformat(),
        "closed_trades": summary["count"],
        "win_rate": summary["win_rate"],
        "avg_pnl": summary["avg_pnl"],
        "expectancy": summary["expectancy"],
        "gross_wins": summary["gross_wins"],
        "gross_losses": summary["gross_losses"],
        "by_source": by_source,
        "by_regime": by_regime,
        "by_symbol": by_symbol,
        "by_hour": by_hour,
        "health": health,
        "loss_streak": int(state.get("meta", {}).get("consecutive_losses", 0)),
        "reconciliation": state.get("meta", {}).get("reconciliation", {}),
    }


def compute_trend_ok(config: BotConfig, htf_closes: List[float]) -> Tuple[bool, float, float, float]:
    fast = sma(htf_closes, config.htf_fast)
    slow = sma(htf_closes, config.htf_slow)
    slow_prev = sum(htf_closes[-config.htf_slow - config.htf_slope_n : -config.htf_slope_n]) / float(config.htf_slow)
    slope = (slow - slow_prev) / float(config.htf_slope_n)
    last_close = htf_closes[-1]

    trend_ok = fast > slow and slope > 0 and last_close > slow
    if config.trend_mode == "loose":
        trend_ok = trend_ok or (last_close > slow and slope > 0)
    elif config.trend_mode == "override":
        trend_ok = trend_ok or last_close >= max(htf_closes[-config.htf_slow :])

    return trend_ok, fast, slow, slope


def volume_ratio(volumes: List[float], window: int) -> float:
    if len(volumes) < window + 1:
        return float("nan")
    baseline = sum(volumes[-window - 1 : -1]) / float(window)
    if baseline <= 0:
        return 1.0
    return volumes[-1] / baseline


def trailing_return(seq: List[float], lookback: int) -> float:
    if len(seq) <= lookback or seq[-lookback - 1] <= 0:
        return 0.0
    return (seq[-1] / seq[-lookback - 1]) - 1.0


def relative_strength_score(symbol_closes: List[float], benchmark_closes: Optional[List[float]], lookback: int) -> float:
    if benchmark_closes is None:
        return 0.0
    return trailing_return(symbol_closes, lookback) - trailing_return(benchmark_closes, lookback)


def estimate_execution_quality(symbol: str, atr_pct: float, vol_ratio: float) -> float:
    spread_bps = SPREAD_BPS_MAP.get(symbol, 12.0)
    spread_penalty = min(0.25, spread_bps / 100.0)
    vol_bonus = 0.0 if not math.isfinite(vol_ratio) else min(0.18, max(-0.10, (vol_ratio - 1.0) * 0.18))
    atr_penalty = min(0.20, max(0.0, (atr_pct - 0.03) * 4.0))
    quality = 0.92 - spread_penalty + vol_bonus - atr_penalty
    return max(0.35, min(1.0, quality))


def build_entry_signal(
    config: BotConfig,
    closes: List[float],
    highs: List[float],
    lows: List[float],
    volumes: List[float],
) -> Optional[EntrySignal]:
    if len(closes) < config.warmup_ltf:
        return None

    atr_value = atr(highs, lows, closes, config.atr_len)
    if not math.isfinite(atr_value) or atr_value <= 0:
        return None

    rsi_value = rsi(closes, config.rsi_len)
    fast_ema_values = ema_series(closes[-(config.ltf_fast_ema * 4) :], config.ltf_fast_ema)
    slow_ema_values = ema_series(closes[-(config.ltf_slow_ema * 4) :], config.ltf_slow_ema)
    if not fast_ema_values or not slow_ema_values:
        return None
    fast_ema = fast_ema_values[-1]
    slow_ema = slow_ema_values[-1]
    atr_pct = atr_value / max(closes[-1], 1e-9)
    vol_ratio = volume_ratio(volumes, config.volume_window)

    breakout_raw, _ = donchian(closes[:-1], config.donchian)
    if not math.isfinite(breakout_raw):
        return None
    breakout_level = breakout_raw * (1.0 + config.breakout_buffer_bps / 10_000.0)
    extension_atr = (closes[-1] - breakout_level) / max(atr_value, 1e-9)
    slope_lookback = min(config.ema_slope_lookback, len(fast_ema_values) - 1)
    fast_ema_slope = fast_ema - fast_ema_values[-1 - slope_lookback]
    ema_spread_pct = (fast_ema - slow_ema) / max(closes[-1], 1e-9)

    momentum_ok = fast_ema > slow_ema and rsi_value >= config.rsi_entry_min
    slope_ok = fast_ema_slope > 0
    volatility_ok = atr_pct >= config.min_atr_pct
    volume_ok = (not math.isfinite(vol_ratio)) or vol_ratio >= config.min_volume_ratio
    breakout_ok = closes[-2] < breakout_level <= closes[-1]
    extension_ok = extension_atr <= config.max_breakout_atr_extension
    rsi_ok = rsi_value <= config.max_entry_rsi

    if not (momentum_ok and slope_ok and volatility_ok and volume_ok and breakout_ok and extension_ok and rsi_ok):
        return None

    stop = breakout_level - config.stop_atr * atr_value
    tp1 = breakout_level + config.tp1_atr * atr_value
    tp2 = breakout_level + config.tp2_atr * atr_value
    trail_anchor = closes[-1] - config.trail_atr * atr_value
    confidence = 1.0
    confidence += min(0.20, max(0.0, ema_spread_pct * 100.0 * 0.25))
    confidence += min(0.15, max(0.0, (vol_ratio - config.min_volume_ratio) * 0.25))
    confidence += min(0.10, max(0.0, (config.max_entry_rsi - rsi_value) / max(config.max_entry_rsi, 1e-9)))
    confidence = max(0.60, min(1.00, confidence))
    return EntrySignal(
        breakout_level=breakout_level,
        stop=stop,
        tp1=tp1,
        tp2=tp2,
        trail_anchor=trail_anchor,
        atr_value=atr_value,
        volume_ratio=vol_ratio,
        rsi_value=rsi_value,
        source="technical",
        confidence=confidence,
        atr_pct=atr_pct,
        ema_spread_pct=ema_spread_pct,
        reason="breakout+momentum",
    )


def build_sentiment_entry_signal(
    config: BotConfig,
    closes: List[float],
    highs: List[float],
    lows: List[float],
    volumes: List[float],
) -> Optional[EntrySignal]:
    if len(closes) < config.warmup_ltf:
        return None

    atr_value = atr(highs, lows, closes, config.atr_len)
    if not math.isfinite(atr_value) or atr_value <= 0:
        return None

    rsi_value = rsi(closes, config.rsi_len)
    vol_ratio = volume_ratio(volumes, config.volume_window)
    last = closes[-1]
    atr_pct = atr_value / max(last, 1e-9)
    if atr_pct < config.min_atr_pct:
        return None
    if rsi_value > config.max_entry_rsi:
        return None

    stop = last - config.stop_atr * atr_value
    tp1 = last + config.tp1_atr * atr_value
    tp2 = last + config.tp2_atr * atr_value
    trail_anchor = last - config.trail_atr * atr_value
    return EntrySignal(
        breakout_level=last,
        stop=stop,
        tp1=tp1,
        tp2=tp2,
        trail_anchor=trail_anchor,
        atr_value=atr_value,
        volume_ratio=vol_ratio,
        rsi_value=rsi_value,
        source="sentiment",
        confidence=0.85,
        atr_pct=atr_pct,
        ema_spread_pct=0.0,
        reason="sentiment_momentum",
    )


def build_news_momentum_signal(
    config: BotConfig,
    closes: List[float],
    highs: List[float],
    lows: List[float],
    volumes: List[float],
    snapshot: Optional[SentimentSnapshot],
) -> Optional[EntrySignal]:
    if snapshot is None:
        return None
    if snapshot.score < config.sentiment_buy_threshold:
        return None
    if snapshot.acceleration < config.news_momentum_min_acceleration:
        return None

    now = now_utc()
    recent_items = [
        item
        for item in snapshot.items
        if item.published_at is not None
        and (now - item.published_at).total_seconds() <= config.news_momentum_max_age_hours * 3600
    ]
    if len(recent_items) < config.news_momentum_min_recent_items:
        return None

    severe_negative_tags = {"hack", "exploit", "lawsuit", "investigation", "delisting", "liquidation"}
    if any(tag in severe_negative_tags for item in recent_items for tag in item.event_tags):
        return None

    catalyst_tags = {"approval", "etf", "partnership", "listing", "upgrade", "launch"}
    catalyst_count = sum(1 for item in recent_items for tag in item.event_tags if tag in catalyst_tags)
    recent_weighted_score = sum(item.score * item.weight for item in recent_items) / max(
        sum(item.weight for item in recent_items),
        1e-9,
    )
    if catalyst_count <= 0 and recent_weighted_score < config.sentiment_buy_threshold + 0.05:
        return None

    base_signal = build_sentiment_entry_signal(config, closes, highs, lows, volumes)
    if base_signal is None:
        return None

    recent_share = len(recent_items) / max(snapshot.sample_size, 1)
    confidence = 0.74
    confidence += min(0.10, max(0.0, snapshot.score - config.sentiment_buy_threshold) * 0.4)
    confidence += min(0.08, max(0.0, snapshot.acceleration - config.news_momentum_min_acceleration) * 0.6)
    confidence += min(0.05, catalyst_count * 0.02)
    confidence += min(0.05, recent_share * 0.05)
    return EntrySignal(
        breakout_level=base_signal.breakout_level,
        stop=base_signal.stop,
        tp1=base_signal.tp1,
        tp2=base_signal.tp2,
        trail_anchor=base_signal.trail_anchor,
        atr_value=base_signal.atr_value,
        volume_ratio=base_signal.volume_ratio,
        rsi_value=base_signal.rsi_value,
        source="news_momentum",
        confidence=max(0.65, min(0.94, confidence)),
        atr_pct=base_signal.atr_pct,
        ema_spread_pct=base_signal.ema_spread_pct,
        reason="fresh_news_catalyst",
    )


def build_pullback_entry_signal(
    config: BotConfig,
    closes: List[float],
    highs: List[float],
    lows: List[float],
    volumes: List[float],
) -> Optional[EntrySignal]:
    if len(closes) < config.warmup_ltf:
        return None
    atr_value = atr(highs, lows, closes, config.atr_len)
    if not math.isfinite(atr_value) or atr_value <= 0:
        return None
    fast_ema_values = ema_series(closes[-(config.ltf_fast_ema * 4) :], config.ltf_fast_ema)
    slow_ema_values = ema_series(closes[-(config.ltf_slow_ema * 4) :], config.ltf_slow_ema)
    if not fast_ema_values or not slow_ema_values:
        return None
    fast_ema = fast_ema_values[-1]
    slow_ema = slow_ema_values[-1]
    prev_fast = fast_ema_values[-2]
    last = closes[-1]
    prev_close = closes[-2]
    rsi_value = rsi(closes, config.rsi_len)
    atr_pct = atr_value / max(last, 1e-9)
    vol_ratio = volume_ratio(volumes, config.volume_window)
    pullback_distance = abs(prev_close - fast_ema) / max(atr_value, 1e-9)
    reclaimed_fast = prev_close <= prev_fast and last > fast_ema
    if not (
        fast_ema > slow_ema
        and last > slow_ema
        and reclaimed_fast
        and pullback_distance <= max(0.8, config.pullback_ema_buffer_atr + 0.4)
        and rsi_value >= max(48.0, config.rsi_entry_min - 6.0)
        and rsi_value <= config.max_entry_rsi
        and atr_pct >= config.min_atr_pct
    ):
        return None
    stop = min(lows[-3:]) - (0.5 * atr_value)
    entry = last
    confidence = 0.72
    confidence += min(0.12, max(0.0, (1.4 - pullback_distance) * 0.08))
    confidence += min(0.08, max(0.0, (vol_ratio - 1.0) * 0.10))
    confidence = max(0.55, min(0.90, confidence))
    return EntrySignal(
        breakout_level=entry,
        stop=stop,
        tp1=entry + config.tp1_atr * atr_value,
        tp2=entry + config.tp2_atr * atr_value,
        trail_anchor=entry - config.trail_atr * atr_value,
        atr_value=atr_value,
        volume_ratio=vol_ratio,
        rsi_value=rsi_value,
        source="pullback",
        confidence=confidence,
        atr_pct=atr_pct,
        ema_spread_pct=(fast_ema - slow_ema) / max(last, 1e-9),
        reason="trend_pullback_reclaim",
    )


def build_compression_breakout_signal(
    config: BotConfig,
    closes: List[float],
    highs: List[float],
    lows: List[float],
    volumes: List[float],
) -> Optional[EntrySignal]:
    if len(closes) < max(config.warmup_ltf, config.compression_window + config.atr_len + 5):
        return None
    atr_value = atr(highs, lows, closes, config.atr_len)
    if not math.isfinite(atr_value) or atr_value <= 0:
        return None
    recent_atrs = []
    for idx in range(len(closes) - config.compression_window, len(closes)):
        window_atr = atr(highs[: idx + 1], lows[: idx + 1], closes[: idx + 1], config.atr_len)
        if math.isfinite(window_atr):
            recent_atrs.append(window_atr)
    if len(recent_atrs) < max(3, config.compression_window // 2):
        return None
    avg_recent_atr = sum(recent_atrs) / len(recent_atrs)
    if atr_value > avg_recent_atr * config.compression_atr_ratio:
        return None
    breakout = build_entry_signal(config, closes, highs, lows, volumes)
    if breakout is None:
        return None
    confidence = min(0.95, breakout.confidence + 0.05)
    return EntrySignal(
        breakout_level=breakout.breakout_level,
        stop=breakout.stop,
        tp1=breakout.tp1,
        tp2=breakout.tp2,
        trail_anchor=breakout.trail_anchor,
        atr_value=breakout.atr_value,
        volume_ratio=breakout.volume_ratio,
        rsi_value=breakout.rsi_value,
        source="compression",
        confidence=confidence,
        atr_pct=breakout.atr_pct,
        ema_spread_pct=breakout.ema_spread_pct,
        reason="volatility_compression_breakout",
    )


def build_failed_breakdown_signal(
    config: BotConfig,
    closes: List[float],
    highs: List[float],
    lows: List[float],
    volumes: List[float],
) -> Optional[EntrySignal]:
    if len(closes) < max(config.warmup_ltf, config.reversal_lookback + config.atr_len + 5):
        return None
    atr_value = atr(highs, lows, closes, config.atr_len)
    if not math.isfinite(atr_value) or atr_value <= 0:
        return None
    breakdown_hi, breakdown_lo = donchian(closes[:-2], config.reversal_lookback)
    if not math.isfinite(breakdown_lo):
        return None
    prev_close = closes[-2]
    last = closes[-1]
    rsi_value = rsi(closes, config.rsi_len)
    vol_ratio = volume_ratio(volumes, config.volume_window)
    failed_breakdown = prev_close < breakdown_lo and last > breakdown_lo and last > prev_close
    if not failed_breakdown:
        return None
    if rsi_value < max(46.0, config.rsi_entry_min - 8.0) or rsi_value > config.max_entry_rsi:
        return None
    stop = min(lows[-3:]) - (0.35 * atr_value)
    confidence = 0.68 + min(0.10, max(0.0, (vol_ratio - 1.0) * 0.12))
    return EntrySignal(
        breakout_level=last,
        stop=stop,
        tp1=last + config.tp1_atr * atr_value,
        tp2=last + config.tp2_atr * atr_value,
        trail_anchor=last - config.trail_atr * atr_value,
        atr_value=atr_value,
        volume_ratio=vol_ratio,
        rsi_value=rsi_value,
        source="reversal",
        confidence=max(0.55, min(0.88, confidence)),
        atr_pct=atr_value / max(last, 1e-9),
        ema_spread_pct=0.0,
        reason="failed_breakdown_reversal",
    )


def score_setup(
    signal: EntrySignal,
    trend_value: float,
    sentiment_snapshot: Optional[SentimentSnapshot],
    regime: RegimeSnapshot,
    relative_strength: float = 0.0,
    execution_quality: float = 1.0,
    cross_asset_factor: float = 1.0,
) -> float:
    sentiment_score = 0.0 if sentiment_snapshot is None else sentiment_snapshot.score
    acceleration = 0.0 if sentiment_snapshot is None else sentiment_snapshot.acceleration
    source_bonus = {
        "technical": 0.10,
        "pullback": 0.12,
        "sentiment": 0.08,
        "compression": 0.14,
        "reversal": 0.10,
        "news_momentum": 0.16,
    }.get(signal.source, 0.0)
    base = (
        signal.confidence
        + source_bonus
        + min(0.25, max(0.0, trend_value / 10.0))
        + min(0.20, max(0.0, sentiment_score * 0.5))
        + min(0.10, max(0.0, acceleration * 0.4))
        + min(0.18, max(-0.12, relative_strength * 1.8))
    )
    return base * regime.risk_multiplier * execution_quality * cross_asset_factor


def sentiment_allows_entry(config: BotConfig, snapshot: Optional[SentimentSnapshot]) -> bool:
    if not config.sentiment_enabled or config.sentiment_mode == "disabled":
        return True
    if config.sentiment_mode != "confirm":
        return True
    return snapshot is not None and snapshot.score >= config.sentiment_buy_threshold


def sentiment_triggers_primary_entry(config: BotConfig, snapshot: Optional[SentimentSnapshot]) -> bool:
    return (
        config.sentiment_enabled
        and config.sentiment_mode == "primary"
        and snapshot is not None
        and snapshot.score >= config.sentiment_buy_threshold
    )


def sentiment_triggers_exit(config: BotConfig, snapshot: Optional[SentimentSnapshot]) -> bool:
    return (
        config.sentiment_enabled
        and config.sentiment_exit_on_bearish
        and snapshot is not None
        and snapshot.score <= config.sentiment_sell_threshold
    )


def can_open_new_risk(config: BotConfig, state: Dict[str, Any]) -> bool:
    if state.get("daily", {}).get("halt", False):
        return False
    if state.get("weekly", {}).get("halt", False):
        return False
    if int(state.get("meta", {}).get("consecutive_losses", 0)) >= config.max_consecutive_losses:
        return False
    return True


def reconcile_positions_with_state(
    config: BotConfig,
    state: Dict[str, Any],
    symbols: List[str],
    positions: Dict[str, BrokerPosition],
) -> Dict[str, int]:
    reconciled = {"restored": 0, "cleared": 0, "mismatched_qty": 0}
    for symbol in symbols:
        sym_state = get_sym_state(state, symbol)
        position = positions.get(symbol)
        snapshot = sym_state.get("snapshot")
        if position is None or position.qty <= 0:
            had_live_state = any(
                sym_state.get(key) is not None
                for key in ("last_entry_price", "high_water", "live_stop", "entry_source", "entry_regime")
            ) or bool(sym_state.get("tp1_hit", False))
            if had_live_state:
                reset_trade_state(sym_state)
                reconciled["cleared"] += 1
                if snapshot is not None:
                    snapshot["last_action"] = "reconciled flat"
                log_line(config, f"RECONCILE {symbol}: cleared stale position state because broker is flat.")
            continue

        last_entry_price = safe_float(sym_state.get("last_entry_price"))
        if last_entry_price <= 0:
            sym_state["last_entry_price"] = position.avg_entry_price
            sym_state["high_water"] = max(
                safe_float(sym_state.get("high_water"), position.current_price),
                position.current_price,
            )
            sym_state["entry_source"] = sym_state.get("entry_source") or "broker_restore"
            sym_state["entry_regime"] = sym_state.get("entry_regime") or "unknown"
            reconciled["restored"] += 1
            if snapshot is not None:
                snapshot["last_action"] = "reconciled live"
            log_line(config, f"RECONCILE {symbol}: restored missing entry state from broker position.")
        elif abs(last_entry_price - position.avg_entry_price) / max(position.avg_entry_price, 1e-9) > 0.02:
            sym_state["last_entry_price"] = position.avg_entry_price
            reconciled["mismatched_qty"] += 1
            if snapshot is not None:
                snapshot["last_action"] = "reconciled avg"
            log_line(config, f"RECONCILE {symbol}: updated entry price to broker average.")
    state.setdefault("meta", {})["reconciliation"] = reconciled
    return reconciled


def execute_ranked_entries(
    config: BotConfig,
    broker: AlpacaPaperBroker,
    state: Dict[str, Any],
    account: AccountSnapshot,
    positions: Dict[str, BrokerPosition],
    prices_now: Dict[str, float],
    candidates: List[CandidateSetup],
    open_order_symbols: Set[str],
) -> None:
    if not candidates or not can_open_new_risk(config, state):
        return

    ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
    allocated = total_notional(positions, prices_now)
    heat_now = portfolio_heat(positions, state, prices_now, account.equity)
    sector_now = sector_exposure(positions, prices_now, account.equity)
    entries_taken = 0

    for candidate in ranked:
        if entries_taken >= config.max_entries_per_loop:
            break
        symbol = candidate.symbol
        if symbol in open_order_symbols:
            continue
        position = positions.get(symbol)
        if position is not None and position.qty > 0:
            continue
        sym_state = get_sym_state(state, symbol)
        if not can_fire(config, sym_state):
            continue

        buy_qty = size_for_risk(
            config,
            account.cash,
            candidate.signal.breakout_level,
            candidate.signal.stop,
            account.equity,
            allocated,
        )
        if buy_qty > 0:
            buy_qty *= (
                candidate.signal.confidence
                * candidate.regime.risk_multiplier
                * candidate.execution_quality
                * candidate.cross_asset_multiplier
            )
        if buy_qty <= 0:
            set_snapshot_candidate_score(state, symbol, candidate.score)
            state["sym"][symbol]["snapshot"]["last_action"] = "size zero"
            continue

        proposed_notional = buy_qty * candidate.signal.breakout_level
        proposed_risk = max(0.0, candidate.signal.breakout_level - candidate.signal.stop) * buy_qty
        if heat_now + (proposed_risk / max(account.equity, 1e-9)) > config.max_portfolio_heat:
            state["sym"][symbol]["snapshot"]["last_action"] = "heat cap"
            write_signal_journal(
                config,
                symbol,
                candidate.signal.source,
                candidate.regime.name,
                candidate.score,
                None if candidate.sentiment is None else candidate.sentiment.score,
                candidate.trend_score,
                "rejected",
                "portfolio_heat",
            )
            continue
        if sector_now.get(candidate.sector, 0.0) + (proposed_notional / max(account.equity, 1e-9)) > config.max_sector_exposure:
            state["sym"][symbol]["snapshot"]["last_action"] = "sector cap"
            write_signal_journal(
                config,
                symbol,
                candidate.signal.source,
                candidate.regime.name,
                candidate.score,
                None if candidate.sentiment is None else candidate.sentiment.score,
                candidate.trend_score,
                "rejected",
                "sector_exposure",
            )
            continue

        order = broker.submit_market_buy(symbol, buy_qty)
        sym_state["tp1_hit"] = False
        sym_state["high_water"] = candidate.last_price
        sym_state["last_entry_price"] = candidate.signal.breakout_level
        sym_state["entry_source"] = candidate.signal.source
        sym_state["entry_regime"] = candidate.regime.name
        sym_state["live_stop"] = candidate.signal.stop
        set_fired(sym_state)
        entries_taken += 1
        allocated += proposed_notional
        heat_now += proposed_risk / max(account.equity, 1e-9)
        sector_now[candidate.sector] = sector_now.get(candidate.sector, 0.0) + (proposed_notional / max(account.equity, 1e-9))
        state.setdefault("meta", {})["entries_this_loop"] = entries_taken

        note = (
            f"order_id={order.id};source={candidate.signal.source};confidence={candidate.signal.confidence:.2f};"
            f"score={candidate.score:.3f};rsi={candidate.signal.rsi_value:.2f};vol_ratio={candidate.signal.volume_ratio:.2f};"
            f"atr={candidate.signal.atr_value:.6f};atr_pct={candidate.signal.atr_pct:.4f};"
            f"ema_spread_pct={candidate.signal.ema_spread_pct:.4f};regime={candidate.regime.name};"
            f"reason={candidate.signal.reason};exec_q={candidate.execution_quality:.3f};rs={candidate.relative_strength:.4f};"
            f"cross_asset={candidate.cross_asset_multiplier:.3f}"
        )
        if candidate.sentiment is not None:
            note += f";sentiment={candidate.sentiment.score:.3f};sentiment_accel={candidate.sentiment.acceleration:.3f}"
        write_trade(
            config,
            symbol,
            "ENTRY_SUBMITTED",
            candidate.last_price,
            candidate.signal.breakout_level,
            candidate.signal.stop,
            candidate.signal.tp1,
            candidate.signal.tp2,
            buy_qty,
            note,
        )
        write_signal_journal(
            config,
            symbol,
            candidate.signal.source,
            candidate.regime.name,
            candidate.score,
            None if candidate.sentiment is None else candidate.sentiment.score,
            candidate.trend_score,
            "accepted",
            candidate.signal.reason,
        )
        snapshot = state.get("sym", {}).get(symbol, {}).get("snapshot")
        if snapshot is not None:
            snapshot["last_action"] = f"buy {candidate.signal.source}"
            snapshot["candidate_score"] = candidate.score
            snapshot["setup_label"] = f"{candidate.signal.source} {candidate.signal.confidence:.2f}"
            snapshot["reason"] = candidate.signal.reason
        log_line(config, f"{symbol} ENTRY submitted qty={buy_qty:.8f} source={candidate.signal.source} rank_score={candidate.score:.3f}")


def compute_live_exit_levels(
    config: BotConfig,
    position: BrokerPosition,
    sym_state: Dict[str, Any],
    closes: List[float],
    highs: List[float],
    lows: List[float],
) -> Tuple[float, float, float]:
    atr_value = atr(highs, lows, closes, config.atr_len)
    entry = position.avg_entry_price
    initial_stop = entry - config.stop_atr * atr_value
    tp1 = entry + config.tp1_atr * atr_value
    tp2 = entry + config.tp2_atr * atr_value

    high_water = max(safe_float(sym_state.get("high_water"), position.current_price), closes[-1], position.current_price)
    sym_state["high_water"] = high_water

    trailing_stop = high_water - config.trail_atr * atr_value
    if sym_state.get("tp1_hit", False):
        stop = max(initial_stop, entry, trailing_stop)
    else:
        stop = max(initial_stop, trailing_stop * 0.98)
    return stop, tp1, tp2


def update_daily_drawdown_halt(config: BotConfig, state: Dict[str, Any]) -> None:
    if config.daily_dd_limit > 0.99:
        return
    start_equity = safe_float(state.get("daily", {}).get("start_equity"), state["equity"])
    current_drawdown = (start_equity - state["equity"]) / max(1e-9, start_equity)
    if current_drawdown >= config.daily_dd_limit:
        state["daily"]["halt"] = True


def update_weekly_drawdown_halt(config: BotConfig, state: Dict[str, Any]) -> None:
    if config.weekly_dd_limit > 0.99:
        return
    start_equity = safe_float(state.get("weekly", {}).get("start_equity"), state["equity"])
    current_drawdown = (start_equity - state["equity"]) / max(1e-9, start_equity)
    if current_drawdown >= config.weekly_dd_limit:
        state["weekly"]["halt"] = True


def process_symbol(
    config: BotConfig,
    data_client: AlpacaCryptoDataClient,
    broker: AlpacaPaperBroker,
    sentiment_client: SentimentClient,
    state: Dict[str, Any],
    symbol: str,
    started_at: datetime,
    account: AccountSnapshot,
    positions: Dict[str, BrokerPosition],
    open_order_symbols: Set[str],
    prices_now: Dict[str, float],
    benchmark_context: Dict[str, List[float]],
    cross_asset: CrossAssetContext,
) -> Optional[CandidateSetup]:
    sym_state = get_sym_state(state, symbol)
    position = positions.get(symbol)
    action_label = "watch"

    htf_closes = data_client.fetch_htf_closes(symbol, max(config.warmup_htf, config.htf_slow + config.htf_slope_n + 5))
    if len(htf_closes) < max(config.htf_slow + config.htf_slope_n, config.warmup_htf):
        log_line(config, f"Not enough HTF bars for {symbol}")
        sym_state["warm_htf_ok"] = False
        return None
    sym_state["warm_htf_ok"] = True

    highs, lows, closes, volumes = data_client.fetch_ltf_ohlcv(symbol, config.warmup_ltf + 5)
    if len(closes) < config.warmup_ltf:
        log_line(config, f"Not enough LTF bars for {symbol}")
        sym_state["warm_ltf_ok"] = False
        return None
    sym_state["warm_ltf_ok"] = True

    last_tick = data_client.fetch_last_price(symbol)
    last = closes[-1] if not math.isfinite(last_tick) else last_tick
    prices_now[symbol] = last

    trend_ok, htf_fast, htf_slow, htf_slope = compute_trend_ok(config, htf_closes)
    trend_value = trend_score(htf_fast, htf_slow, htf_slope, htf_closes[-1])
    log_line(
        config,
        f"HTF {symbol}: fast={round(htf_fast, 2)}, slow={round(htf_slow, 2)}, slope={round(htf_slope, 4)}, score={round(trend_value, 2)}",
    )

    atr_live = atr(highs, lows, closes, config.atr_len)
    atr_pct_live = atr_live / max(closes[-1], 1e-9) if math.isfinite(atr_live) and atr_live > 0 else 0.0
    vol_ratio_live = volume_ratio(volumes, config.volume_window)
    btc_benchmark = benchmark_context.get("BTC/USD")
    eth_benchmark = benchmark_context.get("ETH/USD")
    benchmark = btc_benchmark if symbol != "BTC/USD" else eth_benchmark
    rel_strength = relative_strength_score(closes, benchmark, config.rs_lookback)
    execution_quality = estimate_execution_quality(symbol, atr_pct_live, vol_ratio_live)
    symbol_cross_asset_factor = cross_asset_multiplier(symbol, cross_asset) * cross_asset.risk_multiplier
    breakout_signal = build_entry_signal(config, closes, highs, lows, volumes)
    pullback_signal = build_pullback_entry_signal(config, closes, highs, lows, volumes)
    compression_signal = build_compression_breakout_signal(config, closes, highs, lows, volumes)
    reversal_signal = build_failed_breakdown_signal(config, closes, highs, lows, volumes)
    entry_signal = breakout_signal
    sentiment_snapshot: Optional[SentimentSnapshot] = None
    if sentiment_client.is_active():
        try:
            sentiment_snapshot = sentiment_client.get_sentiment(symbol, config.request_timeout_secs)
        except Exception as exc:
            log_line(config, f"{symbol} sentiment fetch failed: {type(exc).__name__}: {exc}")
        else:
            if sentiment_snapshot is None:
                log_line(config, f"{symbol} sentiment unavailable or below minimum sample.")
            else:
                log_line(
                    config,
                    f"{symbol} sentiment={sentiment_snapshot.score:.3f} "
                    f"label={sentiment_snapshot.label} accel={sentiment_snapshot.acceleration:.3f} "
                    f"samples={sentiment_snapshot.sample_size} sources={sentiment_snapshot.source_counts} "
                    f"events={sentiment_snapshot.event_counts}",
                )

    regime = detect_regime(trend_ok, trend_value, atr_pct_live, vol_ratio_live, sentiment_snapshot)

    if position is None or position.qty <= 0:
        reset_trade_state(sym_state)

    if (now_utc() - started_at).total_seconds() < config.mute_secs:
        action_label = "startup mute"
        update_symbol_snapshot(
            state,
            symbol,
            last=last,
            position=position,
            trend_ok=trend_ok,
            trend_value=trend_value,
            sentiment_snapshot=sentiment_snapshot,
            entry_signal=entry_signal,
            last_action=action_label,
            regime=regime,
            relative_strength=rel_strength,
            execution_quality=execution_quality,
            cross_asset=cross_asset,
        )
        log_line(config, f"{symbol} muted (startup)")
        sym_state["prev_price"] = last
        return None

    update_daily_drawdown_halt(config, state)
    if symbol in open_order_symbols:
        action_label = "open order"
        update_symbol_snapshot(
            state,
            symbol,
            last=last,
            position=position,
            trend_ok=trend_ok,
            trend_value=trend_value,
            sentiment_snapshot=sentiment_snapshot,
            entry_signal=entry_signal,
            last_action=action_label,
            regime=regime,
            relative_strength=rel_strength,
            execution_quality=execution_quality,
            cross_asset=cross_asset,
        )
        log_line(config, f"{symbol} has open orders; waiting.")
        sym_state["prev_price"] = last
        return None

    if position is None or position.qty <= 0:
        setup_options: List[EntrySignal] = []
        if breakout_signal and regime.allow_breakout:
            setup_options.append(breakout_signal)
        if pullback_signal and regime.allow_pullback:
            setup_options.append(pullback_signal)
        if compression_signal and regime.allow_breakout:
            setup_options.append(compression_signal)
        if reversal_signal and regime.allow_trend:
            setup_options.append(reversal_signal)
        news_momentum_signal = build_news_momentum_signal(config, closes, highs, lows, volumes, sentiment_snapshot)
        if news_momentum_signal and (trend_ok or sentiment_triggers_primary_entry(config, sentiment_snapshot)):
            setup_options.append(news_momentum_signal)

        if not trend_ok and regime.allow_trend:
            setup_options = [signal for signal in setup_options if signal.source in {"sentiment", "reversal", "news_momentum"}]

        if not can_open_new_risk(config, state):
            action_label = "halted"
            log_line(config, "Risk halt reached; entries halted.")
        elif not trend_ok:
            if not sentiment_triggers_primary_entry(config, sentiment_snapshot):
                action_label = "trend reject"
                log_line(config, f"{symbol} HTF trend not OK; skip entry.")
            else:
                action_label = "sentiment override"
        else:
            if not sentiment_allows_entry(config, sentiment_snapshot):
                action_label = "sentiment reject"
                update_symbol_snapshot(
                    state,
                    symbol,
                    last=last,
                    position=position,
                    trend_ok=trend_ok,
                    trend_value=trend_value,
                    sentiment_snapshot=sentiment_snapshot,
                    entry_signal=entry_signal,
                    last_action=action_label,
                    regime=regime,
                    relative_strength=rel_strength,
                    execution_quality=execution_quality,
                    cross_asset=cross_asset,
                )
                log_line(config, f"{symbol} sentiment did not confirm long entry.")
                sym_state["prev_price"] = last
                return None

            if sentiment_triggers_primary_entry(config, sentiment_snapshot):
                momentum_signal = build_news_momentum_signal(config, closes, highs, lows, volumes, sentiment_snapshot)
                if momentum_signal is None:
                    momentum_signal = build_sentiment_entry_signal(config, closes, highs, lows, volumes)
                if momentum_signal is not None and all(item.source != momentum_signal.source for item in setup_options):
                    setup_options.append(momentum_signal)
                action_label = "sentiment setup"

        ranked_setups = sorted(
            setup_options,
            key=lambda signal: score_setup(
                signal,
                trend_value,
                sentiment_snapshot,
                regime,
                rel_strength,
                execution_quality,
                symbol_cross_asset_factor,
            ),
            reverse=True,
        )
        entry_signal = ranked_setups[0] if ranked_setups else None

        candidate: Optional[CandidateSetup] = None
        if (
            entry_signal
            and can_fire(config, sym_state)
            and (trend_ok or entry_signal.source in {"sentiment", "reversal", "news_momentum"})
            and rel_strength >= config.min_relative_strength
            and execution_quality >= config.min_execution_quality
        ):
            candidate_score = score_setup(
                entry_signal,
                trend_value,
                sentiment_snapshot,
                regime,
                rel_strength,
                execution_quality,
                symbol_cross_asset_factor,
            )
            candidate = CandidateSetup(
                symbol=symbol,
                signal=entry_signal,
                regime=regime,
                sentiment=sentiment_snapshot,
                trend_score=trend_value,
                sector=sector_for_symbol(symbol),
                score=candidate_score,
                last_price=last,
                execution_quality=execution_quality,
                relative_strength=rel_strength,
                cross_asset_multiplier=symbol_cross_asset_factor,
            )
            set_snapshot_candidate_score(state, symbol, candidate_score)
            action_label = f"candidate {entry_signal.source}"
            write_signal_journal(
                config,
                symbol,
                entry_signal.source,
                regime.name,
                candidate_score,
                None if sentiment_snapshot is None else sentiment_snapshot.score,
                trend_value,
                "candidate",
                entry_signal.reason,
            )
        else:
            if action_label == "watch":
                action_label = "no setup" if rel_strength >= config.min_relative_strength else "weak rs"
            log_line(config, f"{symbol} no valid setup. regime={regime.name}")

        update_symbol_snapshot(
            state,
            symbol,
            last=last,
            position=position,
            trend_ok=trend_ok,
            trend_value=trend_value,
            sentiment_snapshot=sentiment_snapshot,
            entry_signal=entry_signal,
            last_action=action_label,
            regime=regime,
            relative_strength=rel_strength,
            execution_quality=execution_quality,
            cross_asset=cross_asset,
        )
        sym_state["prev_price"] = last
        return candidate

    stop, tp1, tp2 = compute_live_exit_levels(config, position, sym_state, closes, highs, lows)
    sym_state["live_stop"] = stop
    qty = position.qty
    qty_available = position.qty_available

    if (
        sentiment_triggers_exit(config, sentiment_snapshot)
        or (sentiment_snapshot is not None and sentiment_snapshot.score <= config.shock_sentiment_threshold)
    ) and can_fire(config, sym_state) and qty_available > 0:
        order = broker.submit_market_sell(symbol, qty_available)
        set_fired(sym_state)
        action_label = "sell sentiment"
        pnl_value = (last - position.avg_entry_price) * qty_available
        pnl_pct = ((last / max(position.avg_entry_price, 1e-9)) - 1.0) * 100.0
        update_trade_stats(state, symbol, sym_state.get("entry_source", "unknown"), sym_state.get("entry_regime", regime.name), pnl_value, pnl_pct)
        write_trade(
            config,
            symbol,
            "EXIT_SENTIMENT_SUBMITTED",
            last,
            position.avg_entry_price,
            stop,
            tp1,
            tp2,
            qty_available,
            f"order_id={order.id};sentiment={sentiment_snapshot.score:.3f};pnl_value={pnl_value:.4f};pnl_pct={pnl_pct:.3f}",
        )
        log_line(config, f"{symbol} bearish sentiment exit submitted qty={qty_available:.8f}")

    elif (
        not sym_state.get("tp1_hit", False)
        and crossed_up(sym_state.get("prev_price"), last, tp1)
        and can_fire(config, sym_state)
        and qty_available > 0
    ):
        sell_qty = min(qty_available, qty * 0.4)
        if sell_qty > 0:
            order = broker.submit_market_sell(symbol, sell_qty)
            sym_state["tp1_hit"] = True
            set_fired(sym_state)
            action_label = "take profit 1"
            pnl_value = (last - position.avg_entry_price) * sell_qty
            pnl_pct = ((last / max(position.avg_entry_price, 1e-9)) - 1.0) * 100.0
            update_trade_stats(state, symbol, sym_state.get("entry_source", "unknown"), sym_state.get("entry_regime", regime.name), pnl_value, pnl_pct)
            write_trade(config, symbol, "TP1_SUBMITTED", last, position.avg_entry_price, stop, tp1, tp2, sell_qty, f"order_id={order.id};pnl_value={pnl_value:.4f};pnl_pct={pnl_pct:.3f}")
            log_line(config, f"{symbol} TP1 submitted qty={sell_qty:.8f}")

    elif crossed_up(sym_state.get("prev_price"), last, tp2) and can_fire(config, sym_state) and qty_available > 0:
        order = broker.submit_market_sell(symbol, qty_available)
        set_fired(sym_state)
        action_label = "take profit 2"
        pnl_value = (last - position.avg_entry_price) * qty_available
        pnl_pct = ((last / max(position.avg_entry_price, 1e-9)) - 1.0) * 100.0
        update_trade_stats(state, symbol, sym_state.get("entry_source", "unknown"), sym_state.get("entry_regime", regime.name), pnl_value, pnl_pct)
        write_trade(
            config,
            symbol,
            "EXIT_TP2_SUBMITTED",
            last,
            position.avg_entry_price,
            stop,
            tp1,
            tp2,
            qty_available,
            f"order_id={order.id};pnl_value={pnl_value:.4f};pnl_pct={pnl_pct:.3f}",
        )
        log_line(config, f"{symbol} TP2 exit submitted qty={qty_available:.8f}")

    elif crossed_down(sym_state.get("prev_price"), last, stop) and can_fire(config, sym_state) and qty_available > 0:
        order = broker.submit_market_sell(symbol, qty_available)
        set_fired(sym_state)
        action_label = "stop exit"
        pnl_value = (last - position.avg_entry_price) * qty_available
        pnl_pct = ((last / max(position.avg_entry_price, 1e-9)) - 1.0) * 100.0
        update_trade_stats(state, symbol, sym_state.get("entry_source", "unknown"), sym_state.get("entry_regime", regime.name), pnl_value, pnl_pct)
        write_trade(
            config,
            symbol,
            "EXIT_STOP_SUBMITTED",
            last,
            position.avg_entry_price,
            stop,
            tp1,
            tp2,
            qty_available,
            f"order_id={order.id};pnl_value={pnl_value:.4f};pnl_pct={pnl_pct:.3f}",
        )
        log_line(config, f"{symbol} stop exit submitted qty={qty_available:.8f}")
    else:
        action_label = "manage"

    update_symbol_snapshot(
        state,
        symbol,
        last=last,
        position=position,
        trend_ok=trend_ok,
        trend_value=trend_value,
        sentiment_snapshot=sentiment_snapshot,
        entry_signal=entry_signal,
        last_action=action_label,
        regime=regime,
        relative_strength=rel_strength,
        execution_quality=execution_quality,
        cross_asset=cross_asset,
    )
    sym_state["prev_price"] = last
    return None


def run_once(
    config: BotConfig,
    data_client: AlpacaCryptoDataClient,
    broker: AlpacaPaperBroker,
    sentiment_client: SentimentClient,
    state: Dict[str, Any],
    symbols: List[str],
    started_at: datetime,
) -> Dict[str, float]:
    account = broker.get_account_snapshot()
    state["cash"] = account.cash
    state["equity"] = account.equity
    meta = state.setdefault("meta", {})
    meta["last_loop_at"] = now_utc().isoformat()
    meta["entries_this_loop"] = 0
    init_daily_controls(config, state)
    init_weekly_controls(config, state)
    update_loss_streak(state, account.equity)
    update_daily_drawdown_halt(config, state)
    update_weekly_drawdown_halt(config, state)

    positions = broker.get_positions(symbols)
    open_order_symbols = broker.get_open_order_symbols(symbols)
    reconcile_positions_with_state(config, state, symbols, positions)
    prices_now: Dict[str, float] = {}
    candidates: List[CandidateSetup] = []
    benchmark_context: Dict[str, List[float]] = {}
    for benchmark_symbol in ("BTC/USD", "ETH/USD"):
        if benchmark_symbol in symbols:
            try:
                benchmark_context[benchmark_symbol] = data_client.fetch_htf_closes(
                    benchmark_symbol,
                    max(config.warmup_htf, config.htf_slow + config.htf_slope_n + 5),
                )
            except Exception as exc:
                log_line(config, f"{benchmark_symbol} benchmark fetch failed: {type(exc).__name__}: {exc}")
    cross_asset = compute_cross_asset_context(config, benchmark_context)
    meta["cross_asset"] = {
        "regime": cross_asset.regime,
        "btc_trend_score": round(cross_asset.btc_trend_score, 4),
        "eth_trend_score": round(cross_asset.eth_trend_score, 4),
        "eth_vs_btc": round(cross_asset.eth_vs_btc, 4),
    }
    meta["portfolio_heat"] = portfolio_heat(positions, state, prices_now, account.equity)
    meta["sector_exposure"] = sector_exposure(positions, prices_now, account.equity)

    for symbol in symbols:
        try:
            candidate = process_symbol(
                config=config,
                data_client=data_client,
                broker=broker,
                sentiment_client=sentiment_client,
                state=state,
                symbol=symbol,
                started_at=started_at,
                account=account,
                positions=positions,
                open_order_symbols=open_order_symbols,
                prices_now=prices_now,
                benchmark_context=benchmark_context,
                cross_asset=cross_asset,
            )
            if candidate is not None:
                candidates.append(candidate)
        except Exception as exc:
            log_line(config, f"{symbol} ERROR: {type(exc).__name__}: {exc}")

    meta["candidate_count"] = len(candidates)
    meta["no_candidate_loops"] = 0 if candidates else int(meta.get("no_candidate_loops", 0)) + 1
    meta["top_candidates"] = [
        {
            "symbol": candidate.symbol,
            "score": round(candidate.score, 4),
            "setup": candidate.signal.source,
            "regime": candidate.regime.name,
            "cross_asset": round(candidate.cross_asset_multiplier, 3),
        }
        for candidate in sorted(candidates, key=lambda item: item.score, reverse=True)[:5]
    ]
    execute_ranked_entries(config, broker, state, account, positions, prices_now, candidates, open_order_symbols)

    account = broker.get_account_snapshot()
    positions = broker.get_positions(symbols)
    open_order_symbols = broker.get_open_order_symbols(symbols)
    state["cash"] = account.cash
    state["equity"] = account.equity
    meta["portfolio_heat"] = portfolio_heat(positions, state, prices_now, max(account.equity, 1e-9))
    meta["sector_exposure"] = sector_exposure(positions, prices_now, max(account.equity, 1e-9))
    report = build_research_report(state)
    meta["alerts"] = build_runtime_alerts(config, state, report)
    config.research_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_dashboard(config, state, positions, open_order_symbols)
    save_state(config, state)
    log_line(config, f"Heartbeat: equity=${state['equity']:.2f}, cash=${state['cash']:.2f}")
    return prices_now


def main() -> None:
    config = BotConfig.from_env()
    data_client = AlpacaCryptoDataClient(config)
    broker = AlpacaPaperBroker(config)
    sentiment_client = SentimentClient(
        build_session(),
        enabled=config.sentiment_enabled,
        mode=config.sentiment_mode,
        sources=config.sentiment_sources,
        lookback_hours=config.sentiment_lookback_hours,
        min_items=config.sentiment_min_items,
        bullish_threshold=config.sentiment_buy_threshold,
        bearish_threshold=config.sentiment_sell_threshold,
        cache_secs=config.sentiment_cache_secs,
        keyword_map=config.sentiment_keyword_map,
        news_limit=config.sentiment_news_limit,
        twitter_limit=config.sentiment_twitter_limit,
        twitter_rss_url=config.sentiment_twitter_rss_url,
    )
    started_at = now_utc()
    state = load_state(config)

    symbols = [symbol for symbol in config.symbols if data_client.probe_symbol(symbol)]
    ensure_trades_csv(config)
    ensure_signals_csv(config)
    init_daily_controls(config, state)
    init_weekly_controls(config, state)
    log_line(config, f"Active universe: {symbols}")

    while True:
        run_once(config, data_client, broker, sentiment_client, state, symbols, started_at)
        time.sleep(config.loop_secs)


if __name__ == "__main__":
    main()
