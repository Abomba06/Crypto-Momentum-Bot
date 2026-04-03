import math
import time
from datetime import datetime, timedelta, timezone

import requests
from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from app.config import ALPACA_KEY_ID, ALPACA_SECRET_KEY, APCA_API_BASE_URL, FAST, RISK_PCT, SLOW, SYMBOL
from app.risk import clamp_position, notional_from_cash


trading = TradingClient(
    ALPACA_KEY_ID,
    ALPACA_SECRET_KEY,
    paper=True,
    url_override=APCA_API_BASE_URL or None,
)
data = StockHistoricalDataClient(ALPACA_KEY_ID, ALPACA_SECRET_KEY)


def log(message: str) -> None:
    print(f"[{datetime.now()}] {message}")


def market_clock():
    try:
        return trading.get_clock(timeout=10)
    except TypeError:
        return trading.get_clock()


def market_is_open() -> bool:
    return bool(getattr(market_clock(), "is_open", False))


def last_n_closes(n: int = 250):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=n * 2)
    bars = data.get_stock_bars(
        StockBarsRequest(
            symbol_or_symbols=SYMBOL,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed=DataFeed.IEX,
        )
    ).df
    return bars.xs(SYMBOL).close if hasattr(bars, "xs") else bars.close


def sma(series, n: int):
    return series.rolling(n).mean()


def signal() -> str:
    closes = last_n_closes(max(FAST, SLOW) + 5)
    fast_series = sma(closes, FAST)
    slow_series = sma(closes, SLOW)
    fast_now, slow_now = fast_series.iloc[-1], slow_series.iloc[-1]
    fast_prev, slow_prev = fast_series.iloc[-2], slow_series.iloc[-2]

    log(f"Fast SMA={fast_now:.2f}, Slow SMA={slow_now:.2f}")

    if fast_prev <= slow_prev and fast_now > slow_now:
        return "BUY"
    if fast_prev >= slow_prev and fast_now < slow_now:
        return "SELL"
    return "HOLD"


def current_qty(symbol: str = SYMBOL) -> float:
    for position in trading.get_all_positions():
        if position.symbol == symbol:
            return float(position.qty)
    return 0.0


def place_market(side: str, *, notional=None, qty=None) -> None:
    request = MarketOrderRequest(
        symbol=SYMBOL,
        side=OrderSide[side],
        time_in_force=TimeInForce.DAY,
        notional=notional,
        qty=qty,
    )
    trading.submit_order(request)


def run_loop(poll_sec: int = 60, heartbeat_sec: int = 300, max_portfolio_pct: float = 0.20) -> None:
    last_heartbeat = 0.0
    log(f"Bot running for {SYMBOL}. Poll every {poll_sec}s. Paper account.")

    while True:
        now_ts = time.time()

        for attempt in range(3):
            try:
                clock = market_clock()
                break
            except (APIError, requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as exc:
                wait = min(30, 5 * (2 ** attempt))
                log(f"Clock fetch error: {exc}. Retrying in {wait}s...")
                time.sleep(wait)
        else:
            log(f"Clock still unreachable. Sleeping {heartbeat_sec}s...")
            time.sleep(heartbeat_sec)
            continue

        if not getattr(clock, "is_open", False):
            if now_ts - last_heartbeat >= heartbeat_sec:
                log(f"Waiting for market open. next_open={getattr(clock, 'next_open', None)}")
                last_heartbeat = now_ts
            time.sleep(poll_sec)
            continue

        try:
            sig = signal()
        except (APIError, requests.exceptions.RequestException) as exc:
            log(f"Data fetch error: {exc}. Will retry next loop.")
            time.sleep(poll_sec)
            continue

        qty = current_qty()
        try:
            account = trading.get_account()
        except (APIError, requests.exceptions.RequestException) as exc:
            log(f"Account fetch error: {exc}. Will retry next loop.")
            time.sleep(poll_sec)
            continue

        cash, equity = float(account.cash), float(account.equity)

        if sig == "BUY" and qty == 0:
            target = clamp_position(notional_from_cash(cash, RISK_PCT), max_portfolio_pct, equity)
            try:
                place_market("BUY", notional=target)
                log(f"BUY {SYMBOL} ${target}")
            except (APIError, requests.exceptions.RequestException) as exc:
                log(f"Order error (BUY): {exc}")
        elif sig == "SELL" and qty > 0:
            try:
                place_market("SELL", qty=math.floor(qty))
                log(f"SELL {SYMBOL} x{qty}")
            except (APIError, requests.exceptions.RequestException) as exc:
                log(f"Order error (SELL): {exc}")
        else:
            log(f"Decision: {sig}, Qty={qty}, Cash=${cash:.2f}")
            if now_ts - last_heartbeat >= heartbeat_sec:
                log(f"Market open. next_close={getattr(clock, 'next_close', None)}")
                last_heartbeat = now_ts

        time.sleep(poll_sec)


if __name__ == "__main__":
    run_loop()
