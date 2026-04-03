import csv
import json
import math
import os
import pathlib
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


load_dotenv()


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
            state_path=state_path,
            initial_cash=float(os.getenv("INITIAL_CASH", "100000")),
            daily_dd_limit=float(os.getenv("DAILY_DD_LIMIT", "1.0")),
            alpaca_data_url=os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets"),
            apca_key_id=apca_key_id,
            apca_secret_key=apca_secret_key,
            trading_base_url=os.getenv("APCA_API_BASE_URL", "").strip(),
            default_exchange=(os.getenv("ALPACA_CRYPTO_EXCHANGES") or "").strip(),
            request_timeout_secs=int(os.getenv("REQUEST_TIMEOUT_SECS", "20")),
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


def default_state(initial_cash: float) -> Dict[str, Any]:
    return {
        "cash": initial_cash,
        "equity": initial_cash,
        "sym": {},
        "daily": {},
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
        "1h": "1Hour",
        "2h": "2Hour",
        "4h": "4Hour",
        "6h": "6Hour",
        "12h": "12Hour",
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
    fast_ema = ema(closes[-(config.ltf_fast_ema * 3) :], config.ltf_fast_ema)
    slow_ema = ema(closes[-(config.ltf_slow_ema * 3) :], config.ltf_slow_ema)
    atr_pct = atr_value / max(closes[-1], 1e-9)
    vol_ratio = volume_ratio(volumes, config.volume_window)

    breakout_raw, _ = donchian(closes[:-1], config.donchian)
    if not math.isfinite(breakout_raw):
        return None
    breakout_level = breakout_raw * (1.0 + config.breakout_buffer_bps / 10_000.0)

    momentum_ok = fast_ema > slow_ema and rsi_value >= config.rsi_entry_min
    volatility_ok = atr_pct >= config.min_atr_pct
    volume_ok = (not math.isfinite(vol_ratio)) or vol_ratio >= config.min_volume_ratio
    breakout_ok = closes[-2] < breakout_level <= closes[-1]

    if not (momentum_ok and volatility_ok and volume_ok and breakout_ok):
        return None

    stop = breakout_level - config.stop_atr * atr_value
    tp1 = breakout_level + config.tp1_atr * atr_value
    tp2 = breakout_level + config.tp2_atr * atr_value
    trail_anchor = closes[-1] - config.trail_atr * atr_value
    return EntrySignal(
        breakout_level=breakout_level,
        stop=stop,
        tp1=tp1,
        tp2=tp2,
        trail_anchor=trail_anchor,
        atr_value=atr_value,
        volume_ratio=vol_ratio,
        rsi_value=rsi_value,
    )


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


def process_symbol(
    config: BotConfig,
    data_client: AlpacaCryptoDataClient,
    broker: AlpacaPaperBroker,
    state: Dict[str, Any],
    symbol: str,
    started_at: datetime,
    account: AccountSnapshot,
    positions: Dict[str, BrokerPosition],
    open_order_symbols: Set[str],
    prices_now: Dict[str, float],
) -> None:
    sym_state = get_sym_state(state, symbol)
    position = positions.get(symbol)

    htf_closes = data_client.fetch_htf_closes(symbol, max(config.warmup_htf, config.htf_slow + config.htf_slope_n + 5))
    if len(htf_closes) < max(config.htf_slow + config.htf_slope_n, config.warmup_htf):
        log_line(config, f"Not enough HTF bars for {symbol}")
        sym_state["warm_htf_ok"] = False
        return
    sym_state["warm_htf_ok"] = True

    highs, lows, closes, volumes = data_client.fetch_ltf_ohlcv(symbol, config.warmup_ltf + 5)
    if len(closes) < config.warmup_ltf:
        log_line(config, f"Not enough LTF bars for {symbol}")
        sym_state["warm_ltf_ok"] = False
        return
    sym_state["warm_ltf_ok"] = True

    last_tick = data_client.fetch_last_price(symbol)
    last = closes[-1] if not math.isfinite(last_tick) else last_tick
    prices_now[symbol] = last

    trend_ok, htf_fast, htf_slow, htf_slope = compute_trend_ok(config, htf_closes)
    log_line(
        config,
        f"HTF {symbol}: fast={round(htf_fast, 2)}, slow={round(htf_slow, 2)}, slope={round(htf_slope, 4)}",
    )

    entry_signal = build_entry_signal(config, closes, highs, lows, volumes)

    if position is None or position.qty <= 0:
        reset_trade_state(sym_state)

    if (now_utc() - started_at).total_seconds() < config.mute_secs:
        log_line(config, f"{symbol} muted (startup)")
        sym_state["prev_price"] = last
        return

    update_daily_drawdown_halt(config, state)
    if symbol in open_order_symbols:
        log_line(config, f"{symbol} has open orders; waiting.")
        sym_state["prev_price"] = last
        return

    if position is None or position.qty <= 0:
        if not trend_ok:
            log_line(config, f"{symbol} HTF trend not OK; skip entry.")
        elif state.get("daily", {}).get("halt", False):
            log_line(config, "Daily DD limit reached; entries halted.")
        elif entry_signal and can_fire(config, sym_state):
            allocated = total_notional(positions, prices_now)
            buy_qty = size_for_risk(
                config,
                account.cash,
                entry_signal.breakout_level,
                entry_signal.stop,
                account.equity,
                allocated,
            )
            if buy_qty > 0:
                order = broker.submit_market_buy(symbol, buy_qty)
                sym_state["tp1_hit"] = False
                sym_state["high_water"] = last
                sym_state["last_entry_price"] = entry_signal.breakout_level
                set_fired(sym_state)
                note = (
                    f"order_id={order.id};rsi={entry_signal.rsi_value:.2f};"
                    f"vol_ratio={entry_signal.volume_ratio:.2f};atr={entry_signal.atr_value:.6f}"
                )
                write_trade(
                    config,
                    symbol,
                    "ENTRY_SUBMITTED",
                    last,
                    entry_signal.breakout_level,
                    entry_signal.stop,
                    entry_signal.tp1,
                    entry_signal.tp2,
                    buy_qty,
                    note,
                )
                log_line(config, f"{symbol} ENTRY submitted qty={buy_qty:.8f}")
        else:
            log_line(config, f"{symbol} no valid breakout setup.")

        sym_state["prev_price"] = last
        return

    stop, tp1, tp2 = compute_live_exit_levels(config, position, sym_state, closes, highs, lows)
    qty = position.qty
    qty_available = position.qty_available

    if (
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
            write_trade(config, symbol, "TP1_SUBMITTED", last, position.avg_entry_price, stop, tp1, tp2, sell_qty, f"order_id={order.id}")
            log_line(config, f"{symbol} TP1 submitted qty={sell_qty:.8f}")

    elif crossed_up(sym_state.get("prev_price"), last, tp2) and can_fire(config, sym_state) and qty_available > 0:
        order = broker.submit_market_sell(symbol, qty_available)
        set_fired(sym_state)
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
            f"order_id={order.id}",
        )
        log_line(config, f"{symbol} TP2 exit submitted qty={qty_available:.8f}")

    elif crossed_down(sym_state.get("prev_price"), last, stop) and can_fire(config, sym_state) and qty_available > 0:
        order = broker.submit_market_sell(symbol, qty_available)
        set_fired(sym_state)
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
            f"order_id={order.id}",
        )
        log_line(config, f"{symbol} stop exit submitted qty={qty_available:.8f}")

    sym_state["prev_price"] = last


def run_once(
    config: BotConfig,
    data_client: AlpacaCryptoDataClient,
    broker: AlpacaPaperBroker,
    state: Dict[str, Any],
    symbols: List[str],
    started_at: datetime,
) -> Dict[str, float]:
    account = broker.get_account_snapshot()
    state["cash"] = account.cash
    state["equity"] = account.equity
    init_daily_controls(config, state)

    positions = broker.get_positions(symbols)
    open_order_symbols = broker.get_open_order_symbols(symbols)
    prices_now: Dict[str, float] = {}

    for symbol in symbols:
        try:
            process_symbol(
                config=config,
                data_client=data_client,
                broker=broker,
                state=state,
                symbol=symbol,
                started_at=started_at,
                account=account,
                positions=positions,
                open_order_symbols=open_order_symbols,
                prices_now=prices_now,
            )
        except Exception as exc:
            log_line(config, f"{symbol} ERROR: {type(exc).__name__}: {exc}")

    account = broker.get_account_snapshot()
    state["cash"] = account.cash
    state["equity"] = account.equity
    save_state(config, state)
    log_line(config, f"Heartbeat: equity=${state['equity']:.2f}, cash=${state['cash']:.2f}")
    return prices_now


def main() -> None:
    config = BotConfig.from_env()
    data_client = AlpacaCryptoDataClient(config)
    broker = AlpacaPaperBroker(config)
    started_at = now_utc()
    state = load_state(config)

    symbols = [symbol for symbol in config.symbols if data_client.probe_symbol(symbol)]
    ensure_trades_csv(config)
    init_daily_controls(config, state)
    log_line(config, f"Active universe: {symbols}")

    while True:
        run_once(config, data_client, broker, state, symbols, started_at)
        time.sleep(config.loop_secs)


if __name__ == "__main__":
    main()
