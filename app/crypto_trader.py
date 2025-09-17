import os, time, math, csv
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
load_dotenv()


SYMBOLS = [s.strip() for s in os.getenv("CRYPTO_SYMBOLS", "BTC/USD,ETH/USD,SOL/USD").split(",") if s.strip()]

TF_RAW  = os.getenv("CRYPTO_TIMEFRAME", "5Min").strip()
HTF_RAW = os.getenv("HTF_TIMEFRAME", "15Min").strip()

DONCHIAN  = int(os.getenv("DONCHIAN", "10"))
ATR_LEN   = int(os.getenv("ATR_LEN", "10"))
STOP_ATR  = float(os.getenv("STOP_ATR", "1.6"))
TP_MULT   = float(os.getenv("TP_MULT", "1.6"))
TRAIL_ATR = float(os.getenv("TRAIL_ATR", "1.2"))  # for trailing stop

RISK_PCT       = float(os.getenv("RISK_PCT", "0.02"))
PER_COIN_CAP   = float(os.getenv("PER_COIN_CAP", "600"))
PORTFOLIO_CAP  = float(os.getenv("PORTFOLIO_CAP", "0.50"))
COOLDOWN_MIN   = int(os.getenv("COOLDOWN_MIN", "3"))
DAILY_DD_LIMIT = float(os.getenv("DAILY_DD_LIMIT", "0.05"))
REGIME_VOL_MIN = float(os.getenv("REGIME_VOL_MIN", "1.5"))  # ATR% of price
MAX_HOLD_BARS  = int(os.getenv("MAX_HOLD_BARS", "20"))

LOG_DIR = os.getenv("LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
TRADE_LOG = os.path.join(LOG_DIR, "trades_v2.csv")



from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.common.exceptions import APIError
import requests
import pandas as pd

from app.config import ALPACA_KEY_ID, ALPACA_SECRET_KEY

trading = TradingClient(ALPACA_KEY_ID, ALPACA_SECRET_KEY, paper=True)
data    = CryptoHistoricalDataClient()




LAST_QTY         = {s: 0.0 for s in SYMBOLS}
COOLDOWN_UNTIL   = {s: datetime.min.replace(tzinfo=timezone.utc) for s in SYMBOLS}
DAY_START_EQUITY = None
LOSS_TODAY       = {}
POS_STATE        = {s: {"half_taken": False, "entry_ts": None} for s in SYMBOLS}





def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def tf_from_str(s: str) -> TimeFrame:
    s = s.lower()
    if s in ("minute", "1min", "1m"): return TimeFrame.Minute
    if s in ("hour", "1h"):            return TimeFrame.Hour
    if s in ("day", "1d"):             return TimeFrame.Day
    if s.endswith("min"):
        try:
            n = int(s.replace("min", ""))
            return TimeFrame(n, TimeFrameUnit.Minute)
        except Exception:
            return TimeFrame.Minute
    return TimeFrame.Minute

def tf_minutes(tf: TimeFrame) -> int:
    try:
        if hasattr(tf, "amount") and tf.unit == TimeFrameUnit.Minute:
            return int(tf.amount)
    except Exception:
        pass
    if tf == TimeFrame.Minute: return 1
    if tf == TimeFrame.Hour:   return 60
    if tf == TimeFrame.Day:    return 60*24
    return 1

TF  = tf_from_str(TF_RAW)
HTF = tf_from_str(HTF_RAW)
TF_MINUTES = tf_minutes(TF)

def bar_history(symbol: str, timeframe: TimeFrame, lookback_bars: int) -> pd.DataFrame:
    """Fetch OHLCV bars for symbol/timeframe, return last `lookback_bars` rows."""
    end = now_utc()
    if timeframe == TimeFrame.Day:
        start = end - timedelta(days=lookback_bars + 15)
    elif timeframe == TimeFrame.Hour:
        start = end - timedelta(days=max(7, int(lookback_bars/12) + 3))
    else:
        start = end - timedelta(days=3)
    resp = data.get_crypto_bars(CryptoBarsRequest(
        symbol_or_symbols=symbol, timeframe=timeframe, start=start, end=end
    ))
    df = resp.df
    if hasattr(df, "xs"):
        try:
            df = df.xs(symbol)
        except Exception:
            pass
    return df.tail(lookback_bars).copy()

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def donchian(df: pd.DataFrame, n: int):
    return df["high"].rolling(n).max(), df["low"].rolling(n).min()

def get_account():
    return trading.get_account()

def positions_snapshot():
    d = {}
    for p in trading.get_all_positions():
        d[p.symbol] = p
    return d

def qty_for_symbol(symbol: str) -> float:
    wanted = symbol.replace("/", "")
    try:
        for p in trading.get_all_positions():
            if p.symbol in (symbol, wanted):
                LAST_QTY[symbol] = float(p.qty)
                return LAST_QTY[symbol]
        LAST_QTY[symbol] = 0.0
        return 0.0
    except Exception as e:
        print(f"[{now_utc()}] positions error {symbol}: {e}; using cached {LAST_QTY.get(symbol,0.0)}")
        return LAST_QTY.get(symbol, 0.0)

def place_market(symbol: str, side: str, *, notional=None, qty=None, tif: TimeInForce = TimeInForce.GTC):
    req = MarketOrderRequest(
        symbol=symbol, side=OrderSide[side], time_in_force=tif, notional=notional, qty=qty
    )
    return trading.submit_order(req)

def append_trade_log(row: dict):
    write_header = not os.path.exists(TRADE_LOG)
    with open(TRADE_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header: w.writeheader()
        w.writerow(row)

def risk_sized_notional(equity: float) -> float:
    dollar_risk = max(5.0, equity * RISK_PCT)
    return min(PER_COIN_CAP, 4.0 * dollar_risk)

def daily_drawdown_hit(acct) -> bool:
    global DAY_START_EQUITY
    ts = now_utc()
    if DAY_START_EQUITY is None or ts.date() != DAY_START_EQUITY[0].date():
        DAY_START_EQUITY = (ts, float(acct.equity))
        return False
    start_eq = DAY_START_EQUITY[1]
    dd = (float(acct.equity) - start_eq) / max(1e-9, start_eq)
    if dd <= -DAILY_DD_LIMIT:
        print(f"[{now_utc()}] Daily drawdown {dd:.2%} ≤ -{DAILY_DD_LIMIT:.0%}. Pausing until UTC midnight.")
        return True
    return False

def htf_trend_ok(symbol: str) -> bool:
    """Require HTF SMA50 > SMA200 to allow long entries."""
    bars = bar_history(symbol, HTF, 220)
    if len(bars) < 200:
        print(f"[{now_utc()}] {symbol} HTF not enough bars; skip.")
        return False
    s50  = bars["close"].rolling(50).mean().iloc[-1]
    s200 = bars["close"].rolling(200).mean().iloc[-1]
    print(f"[{now_utc()}] HTF trend {symbol}: SMA50={s50:.2f}, SMA200={s200:.2f}")
    return s50 > s200





def manage_symbol(symbol: str, acct, open_positions: dict):
    # Respect cooldown
    if now_utc() < COOLDOWN_UNTIL[symbol]:
        print(f"[{now_utc()}] {symbol} in cooldown until {COOLDOWN_UNTIL[symbol]}.")
        return

    # Get signal-timeframe bars
    lookback = max(ATR_LEN, DONCHIAN) + 5
    try:
        bars = bar_history(symbol, TF, lookback)
    except Exception as e:
        print(f"[{now_utc()}] bar fetch error {symbol}: {e}")
        return
    if len(bars) < lookback - 2:
        print(f"[{now_utc()}] not enough TF bars for {symbol}")
        return
    bars = bars.dropna()

    close = bars["close"]
    last  = float(close.iloc[-1])
    hi, lo = donchian(bars, DONCHIAN)
    upper, lower = float(hi.iloc[-2]), float(lo.iloc[-2])  # previous bar levels
    atr_val = float(atr(bars, ATR_LEN).iloc[-1])

    # Check if we already hold
    qty = qty_for_symbol(symbol)
    wanted = symbol.replace("/", "")
    pos = open_positions.get(symbol) or open_positions.get(wanted)

 
 

    if qty > 0 and pos is not None:
        avg = float(pos.avg_entry_price)
        r   = STOP_ATR * atr_val

        # base stop + trailing stop
        base_stop  = avg - r
        trail_stop = last - TRAIL_ATR * atr_val
        stop = max(base_stop, trail_stop)

        # TPs and time-based exit
        tp1 = avg + r
        tp2 = avg + r * TP_MULT
        ent_ts = POS_STATE[symbol]["entry_ts"]
        max_hold_sec = MAX_HOLD_BARS * TF_MINUTES * 60
        timed_out = ent_ts and (now_utc() - ent_ts).total_seconds() >= max_hold_sec

        print(f"[{now_utc()}] {symbol} qty={qty}, last={last:.6f}, avg={avg:.6f}, "
              f"stop={stop:.6f}, tp1={tp1:.6f}, tp2={tp2:.6f} (ATR={atr_val:.6f})")

        # take 50% at +1R
        if (not POS_STATE[symbol]["half_taken"]) and last >= tp1:
            try:
                sell_qty = qty * 0.5
                place_market(symbol, "SELL", qty=sell_qty, tif=TimeInForce.GTC)
                POS_STATE[symbol]["half_taken"] = True
                append_trade_log({
                    "ts": now_utc().isoformat(), "symbol": symbol, "action": "TP1",
                    "price_last": last, "avg_entry": avg, "qty": sell_qty
                })
                print(f"[{now_utc()}] TP1 hit {symbol}: sold 50% ({sell_qty}).")
                return
            except Exception as e:
                print(f"[{now_utc()}] TP1 order error {symbol}: {e}")

        # exit on stop / tp2 / timeout
        if last <= stop or last >= tp2 or timed_out:
            reason = "stop" if last <= stop else ("tp2" if last >= tp2 else "timeout")
            try:
                place_market(symbol, "SELL", qty=qty, tif=TimeInForce.GTC)
                pnl_est = (last - avg) * qty
                LOSS_TODAY[symbol] = LOSS_TODAY.get(symbol, 0.0) + pnl_est
                if last <= stop and LOSS_TODAY[symbol] <= -(PER_COIN_CAP * 0.5):
                    COOLDOWN_UNTIL[symbol] = now_utc() + timedelta(hours=6)
                    print(f"[{now_utc()}] {symbol} hit per-coin loss limit; cooling 6h.")
                append_trade_log({
                    "ts": now_utc().isoformat(), "symbol": symbol, "action": "EXIT",
                    "price_last": last, "avg_entry": avg, "stop": stop, "tp2": tp2,
                    "qty": qty, "reason": reason
                })
                print(f"[{now_utc()}] EXIT {symbol} at market ({reason}).")
            except Exception as e:
                print(f"[{now_utc()}] order error exit {symbol}: {e}")
        return




    # HTF filter (trend)
    if not htf_trend_ok(symbol):
        print(f"[{now_utc()}] {symbol} HTF trend not OK; skip entry.")
        return

    # Regime filter by volatility
    vol_pct = (atr_val / max(1e-9, last)) * 100.0
    if vol_pct < REGIME_VOL_MIN:
        print(f"[{now_utc()}] {symbol} regime filter: ATR%={vol_pct:.2f} < {REGIME_VOL_MIN:.2f}. Skip.")
        return

    # Donchian breakout: last > prior upper band
    if last > upper:
        equity = float(acct.equity)
        cash   = float(acct.cash)

        deployed = 0.0
        try:
            for p in trading.get_all_positions():
                deployed += float(p.market_value)
        except Exception:
            pass

        if deployed > equity * PORTFOLIO_CAP:
            print(f"[{now_utc()}] portfolio cap reached ({deployed/equity:.2%}). Skip {symbol}.")
            return

        notional = risk_sized_notional(equity)
        notional = min(notional, cash, PER_COIN_CAP)
        if notional < 5.0:
            print(f"[{now_utc()}] insufficient cash for {symbol}: ${notional:.2f}")
            return

        try:
            place_market(symbol, "BUY", notional=round(notional, 2), tif=TimeInForce.GTC)
            POS_STATE[symbol]["half_taken"] = False
            POS_STATE[symbol]["entry_ts"]   = now_utc()
            append_trade_log({
                "ts": now_utc().isoformat(),
                "symbol": symbol,
                "action": "ENTRY",
                "price_last": last,
                "upper": upper,
                "atr": atr_val,
                "notional": notional
            })
            print(f"[{now_utc()}] ENTRY {symbol} BUY ${notional:.2f} (Donchian{DONCHIAN} breakout).")
        except Exception as e:
            print(f"[{now_utc()}] order error entry {symbol}: {e}")
    else:
        print(f"[{now_utc()}] HOLD {symbol} last={last:.6f} upper={upper:.6f}")




def run_loop(poll_sec: int = 60, heartbeat_sec: int = 300):
    last_heart = 0.0
    print(f"[{now_utc()}] V2 crypto bot running on {SYMBOLS} [TF={TF_RAW}, HTF={HTF_RAW}] (paper).")

    while True:
        t0 = time.time()
        # account (with resilience)
        try:
            acct = get_account()
        except (APIError, requests.exceptions.RequestException) as e:
            print(f"[{now_utc()}] account error: {e}; retry next loop")
            time.sleep(poll_sec); continue

        # daily circuit breaker
        if daily_drawdown_hit(acct):
            time.sleep(heartbeat_sec); continue

        # snapshot positions once per loop
        try:
            open_pos = positions_snapshot()
        except Exception as e:
            print(f"[{now_utc()}] positions snapshot error: {e}")
            open_pos = {}

        # manage each symbol
        for sym in SYMBOLS:
            try:
                manage_symbol(sym, acct, open_pos)
            except Exception as e:
                print(f"[{now_utc()}] manage error {sym}: {e}")

        # heartbeat
        if time.time() - last_heart >= heartbeat_sec:
            print(f"[{now_utc()}] Heartbeat: equity=${float(acct.equity):.2f}, cash=${float(acct.cash):.2f}")
            last_heart = time.time()

        # keep loop frequency steady
        sleep_left = poll_sec - (time.time() - t0)
        if sleep_left > 0:
            time.sleep(sleep_left)

if __name__ == "__main__":
    run_loop()
