# app/trader.py for stocks and ELAs 
import math, time
from datetime import datetime, timedelta, timezone

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from alpaca.common.exceptions import APIError
import requests

from app.config import ALPACA_KEY_ID, ALPACA_SECRET_KEY, SYMBOL, FAST, SLOW, RISK_PCT
from app.risk import notional_from_cash, clamp_position


trading = TradingClient(ALPACA_KEY_ID, ALPACA_SECRET_KEY, paper=True)
data    = StockHistoricalDataClient(ALPACA_KEY_ID, ALPACA_SECRET_KEY)


def market_clock():
    """Get Alpaca market clock; tolerate older alpaca-py without timeout kwarg."""
    try:
        return trading.get_clock(timeout=10)  # recent alpaca-py
    except TypeError:
        return trading.get_clock()            # older versions

def market_is_open() -> bool:
    clk = market_clock()
    return bool(getattr(clk, "is_open", False))

def last_n_closes(n=250):
    """Fetch recent daily closes using IEX feed (free/paper)."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=n*2)  # buffer weekends/holidays
    bars = data.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=SYMBOL,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed=DataFeed.IEX   # use IEX instead of SIP
    )).df
    closes = bars.xs(SYMBOL).close if hasattr(bars, "xs") else bars.close
    return closes

def sma(series, n):
    return series.rolling(n).mean()

def signal():
    closes = last_n_closes(max(FAST, SLOW) + 5)
    f_now,  s_now  = sma(closes, FAST).iloc[-1], sma(closes, SLOW).iloc[-1]
    f_prev, s_prev = sma(closes, FAST).iloc[-2], sma(closes, SLOW).iloc[-2]

    # Debug: show SMA values each poll
    print(f"[{datetime.now()}] Fast SMA={f_now:.2f}, Slow SMA={s_now:.2f}")

    if f_prev <= s_prev and f_now > s_now:
        return "BUY"
    if f_prev >= s_prev and f_now < s_now:
        return "SELL"
    return "HOLD"

def current_qty(symbol=SYMBOL) -> float:
    for p in trading.get_all_positions():
        if p.symbol == symbol:
            return float(p.qty)
    return 0.0

def place_market(side, *, notional=None, qty=None):
    req = MarketOrderRequest(
        symbol=SYMBOL,
        side=OrderSide[side],
        time_in_force=TimeInForce.DAY,
        notional=notional,
        qty=qty
    )
    trading.submit_order(req)


def run_loop(poll_sec=60, heartbeat_sec=300, max_portfolio_pct=0.20):
    last_heartbeat = 0.0
    print(f"[{datetime.now()}] Bot running for {SYMBOL}. Poll every {poll_sec}s. Paper account.")

    while True:
        now_ts = time.time()

        for attempt in range(3):
            try:
                clk = market_clock()
                break
            except (APIError, requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                wait = min(30, 5 * (2 ** attempt))  # 5s, 10s, 20s
                print(f"[{datetime.now()}] Clock fetch error: {e}. Retrying in {wait}s…")
                time.sleep(wait)
        else:
            print(f"[{datetime.now()}] Clock still unreachable. Sleeping {heartbeat_sec}s…")
            time.sleep(heartbeat_sec)
            continue

        if not getattr(clk, "is_open", False):
            # Heartbeat while closed
            if now_ts - last_heartbeat >= heartbeat_sec:
                nxt = getattr(clk, "next_open", None)
                print(f"[{datetime.now()}] Waiting for market open… next_open={nxt}")
                last_heartbeat = now_ts
            time.sleep(poll_sec)
            continue

        # Market is open → trade
        try:
            sig = signal()
        except (APIError, requests.exceptions.RequestException) as e:
            print(f"[{datetime.now()}] Data fetch error: {e}. Will retry next loop.")
            time.sleep(poll_sec)
            continue

        qty = current_qty()
        try:
            acct = trading.get_account()
        except (APIError, requests.exceptions.RequestException) as e:
            print(f"[{datetime.now()}] Account fetch error: {e}. Will retry next loop.")
            time.sleep(poll_sec)
            continue

        cash, equity = float(acct.cash), float(acct.equity)

        if sig == "BUY" and qty == 0:
            target = notional_from_cash(cash, RISK_PCT)
            target = clamp_position(target, max_portfolio_pct, equity)
            try:
                place_market("BUY", notional=target)
                print(f"[{datetime.now()}] BUY {SYMBOL} ${target}")
            except (APIError, requests.exceptions.RequestException) as e:
                print(f"[{datetime.now()}] Order error (BUY): {e}")
        elif sig == "SELL" and qty > 0:
            try:
                place_market("SELL", qty=math.floor(qty))
                print(f"[{datetime.now()}] SELL {SYMBOL} x{qty}")
            except (APIError, requests.exceptions.RequestException) as e:
                print(f"[{datetime.now()}] Order error (SELL): {e}")
        else:
            # Always show decision each loop
            print(f"[{datetime.now()}] Decision: {sig}, Qty={qty}, Cash=${cash:.2f}")

            # Light heartbeat during open hours
            if now_ts - last_heartbeat >= heartbeat_sec:
                nxt_close = getattr(clk, "next_close", None)
                print(f"[{datetime.now()}] Market open. NextClose={nxt_close}")
                last_heartbeat = now_ts

        time.sleep(poll_sec)

if __name__ == "__main__":
    run_loop()
