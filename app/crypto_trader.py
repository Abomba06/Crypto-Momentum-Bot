# crypto_trader_v2.py  — Alpaca crypto paper runner
# - Donchian breakout entries on LTF, HTF trend gate (strict/loose/override)
# - ATR stop + TP1/TP2, startup mute, cooldown, daily DD halt
# - Sizing: risk %, PER_COIN_CAP (dollars or fraction), PORTFOLIO_CAP
from dotenv import load_dotenv
import os, time, json, math, csv, pathlib
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Tuple
import requests

load_dotenv()

SYMBOLS     = [s.strip() for s in os.getenv("CRYPTO_SYMBOLS", "BTC/USD,ETH/USD,SOL/USD,DOGE/USD,AVAX/USD,LINK/USD,ADA/USD,MATIC/USD,UNI/USD,LTC/USD,XRP/USD").split(",") if s.strip()]
TF_RAW      = os.getenv("CRYPTO_TIMEFRAME", "5Min").lower()     # LTF (entries/exits)
HTF_RAW     = os.getenv("HTF_TIMEFRAME", "1Day").lower()        # HTF (trend filter)

DONCHIAN    = int(os.getenv("DONCHIAN", "10"))
ATR_LEN     = int(os.getenv("ATR_LEN", "10"))
STOP_ATR    = float(os.getenv("STOP_ATR", "1.6"))
TP_MULT     = float(os.getenv("TP_MULT", "1.6"))
TRAIL_ATR   = float(os.getenv("TRAIL_ATR", "0.0"))

RISK_PCT       = float(os.getenv("RISK_PCT", "0.02"))           # fraction of equity
# PER_COIN_CAP: if >=1 -> dollars cap; if <1 -> fraction of cash
PER_COIN_CAP   = float(os.getenv("PER_COIN_CAP", "0.20"))
PORTFOLIO_CAP  = float(os.getenv("PORTFOLIO_CAP", "1.0"))       # fraction of equity deployed

TREND_MODE     = os.getenv("TREND_MODE", "strict").lower()      # strict | loose | override
WARMUP_LTF     = int(os.getenv("WARMUP_LTF", str(max(DONCHIAN, ATR_LEN)+10)))
WARMUP_HTF     = int(os.getenv("WARMUP_HTF", "205"))            # need >= 200 + slope lookback
HTF_SLOPE_N    = int(os.getenv("HTF_SLOPE_LOOKBACK", "10"))

MUTE_SECS      = int(os.getenv("MUTE_SECS", "90"))
COOLDOWN_SECS  = int(os.getenv("COOLDOWN_SECS", os.getenv("COOLDOWN_MIN", "0")))
if os.getenv("COOLDOWN_MIN"): COOLDOWN_SECS = int(float(os.getenv("COOLDOWN_MIN")))*60
LOOP_SECS      = int(os.getenv("LOOP_SECS", "15"))
DEBUG          = os.getenv("DEBUG", "1") == "1"

BASE = pathlib.Path(".")
LOGS = pathlib.Path(os.getenv("LOG_DIR", "logs"))
LOGS.mkdir(exist_ok=True)
RUN_LOG = LOGS / "run.log"
TRD_CSV = LOGS / "trades.csv"

STATE_PATH   = os.getenv("STATE_PATH", str((BASE / "state.json").resolve()))
STATE_F      = pathlib.Path(STATE_PATH)
INITIAL_CASH = float(os.getenv("INITIAL_CASH", "100000"))
DAILY_DD_LIMIT = float(os.getenv("DAILY_DD_LIMIT", "1.0"))      # 0.05 = stop new entries at -5% day

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def log_line(s: str):
    with RUN_LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{now_utc().isoformat()}] {s}\n")

def ensure_trades_csv():
    header = ["ts","symbol","action","price_last","avg_entry","stop","tp1","tp2","qty","note"]
    new = not TRD_CSV.exists()
    with TRD_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f); 
        if new: w.writerow(header)

def write_trade(symbol, action, price_last, avg_entry, stop, tp1, tp2, qty, note=""):
    ensure_trades_csv()
    with TRD_CSV.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            now_utc().isoformat(), symbol, action, f"{(price_last or 0):.6f}",
            f"{(avg_entry or 0):.6f}", f"{(stop or 0):.6f}",
            f"{(tp1 or 0):.6f}", f"{(tp2 or 0):.6f}", f"{(qty or 0):.8f}", note
        ])

def crossed_up(prev: float, now: float, level: float) -> bool:
    return (prev is not None) and (level is not None) and (prev < level <= now)

def crossed_down(prev: float, now: float, level: float) -> bool:
    return (prev is not None) and (level is not None) and (prev > level >= now)

def sma(seq: List[float], n: int) -> float:
    if len(seq) < n: return float('nan')
    return sum(seq[-n:]) / float(n)

def donchian(seq: List[float], n: int) -> Tuple[float,float]:
    if len(seq) < n: return float('nan'), float('nan')
    window = seq[-n:];  return max(window), min(window)

def true_ranges(highs: List[float], lows: List[float], closes: List[float]) -> List[float]:
    trs = []
    for i in range(1, len(closes)):
        hi, lo, prev_close = highs[i], lows[i], closes[i-1]
        trs.append(max(hi - lo, abs(hi - prev_close), abs(lo - prev_close)))
    return trs

def atr(highs: List[float], lows: List[float], closes: List[float], n: int) -> float:
    if len(closes) < n+1: return float('nan')
    trs = true_ranges(highs, lows, closes)
    if len(trs) < n: return float('nan')
    return sum(trs[-n:]) / float(n)

DEFAULT_STATE = {
    "cash": INITIAL_CASH,
    "equity": INITIAL_CASH,
    "positions": {},   # symbol -> dict
    "sym": {},         # symbol -> per-symbol state
    "daily": {}        # date/start_equity/halt
}

def load_state() -> Dict[str, Any]:
    if STATE_F.exists():
        try:
            s = json.loads(STATE_F.read_text())
            s.setdefault("cash", INITIAL_CASH)
            s.setdefault("equity", s.get("cash", INITIAL_CASH))
            s.setdefault("positions", {})
            s.setdefault("sym", {})
            s.setdefault("daily", {})
            return s
        except Exception as e:
            log_line(f"STATE_LOAD_ERROR: {e!r}")
    return DEFAULT_STATE.copy()

def save_state(st: Dict[str, Any]):
    STATE_F.write_text(json.dumps(st, indent=2))

def get_sym_state(st: Dict[str, Any], sym: str) -> Dict[str, Any]:
    s = st["sym"].get(sym)
    if not s:
        s = {"prev_price": None, "last_fire_ts": None, "tp1_hit": False, "warm_ltf_ok": False, "warm_htf_ok": False}
        st["sym"][sym] = s
    return s

def get_pos(st: Dict[str, Any], sym: str) -> Dict[str, Any]:
    p = st["positions"].get(sym)
    if not p:
        p = {"qty": 0.0, "avg_entry": None, "stop": None, "tp1": None, "tp2": None}
        st["positions"][sym] = p
    return p

def mark_to_market(st: Dict[str, Any], prices: Dict[str, float]):
    eq = st["cash"]
    for sym, p in st["positions"].items():
        last = prices.get(sym)
        if last and p["qty"]:
            eq += p["qty"] * last
    st["equity"] = eq

def can_fire(sym_state: Dict[str, Any]) -> bool:
    ts = now_utc().timestamp()
    last = sym_state.get("last_fire_ts")
    return True if last is None else ((ts - float(last)) >= COOLDOWN_SECS)

def set_fired(sym_state: Dict[str, Any]):
    sym_state["last_fire_ts"] = now_utc().timestamp()

def total_notional(st: Dict[str, Any], price_map: Dict[str, float]) -> float:
    ntl = 0.0
    for sym, p in st["positions"].items():
        if p["qty"] and price_map.get(sym):
            ntl += abs(p["qty"] * price_map[sym])
    return ntl

def today_str(): return now_utc().date().isoformat()

def init_daily_controls(state: Dict[str, Any]):
    d = state.setdefault("daily", {})
    if d.get("date") != today_str():
        d["date"] = today_str()
        d["start_equity"] = state.get("equity", INITIAL_CASH)
        d["halt"] = False

def probe_symbol(sym: str) -> bool:
    try:
        tf = _map_tf_alp(HTF_RAW)          # e.g., "15Min"
        bars = _try_bars(sym, tf, 10)      # tiny probe
        if bars:
            return True
        log_line(f"[WARN] {sym} has 0 bars from Alpaca; dropping from universe.")
    except Exception as e:
        log_line(f"[WARN] {sym} probe failed: {repr(e)}; dropping.")
    return False


import requests
from requests.exceptions import HTTPError

ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
APCA_KEY_ID     = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_KEY_ID", "")
APCA_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY", "")

if not APCA_KEY_ID or not APCA_SECRET_KEY:
    raise RuntimeError("Missing Alpaca API keys (APCA_API_KEY_ID/APCA_API_SECRET_KEY or ALPACA_KEY_ID/ALPACA_SECRET_KEY)")

_headers = {
    "APCA-API-KEY-ID": APCA_KEY_ID,
    "APCA-API-SECRET-KEY": APCA_SECRET_KEY,
}

# Blank by default; if you set ALPACA_CRYPTO_EXCHANGES=CBSE, we will try symbol:CBSE first.
DEFAULT_EXCH = (os.getenv("ALPACA_CRYPTO_EXCHANGES") or "").strip()

_TF_MAP_ALP = {
    "1min": "1Min", "3min": "3Min", "5min": "5Min", "15min": "15Min", "30min": "30Min",
    "1h": "1Hour", "2h": "2Hour", "4h": "4Hour", "6h": "6Hour", "12h": "12Hour",
    "day": "1Day", "1d": "1Day"
}

def _map_tf_alp(tf: str) -> str:
    tl = tf.lower()
    if tl in _TF_MAP_ALP:
        return _TF_MAP_ALP[tl]
    raise ValueError(f"Unsupported timeframe for Alpaca: {tf}")

def _alp_get(path: str, params: dict) -> dict:
    r = requests.get(ALPACA_DATA_URL + path, params=params, headers=_headers, timeout=20)
    try:
        r.raise_for_status()
    except HTTPError as e:
        try:
            msg = r.json()
        except Exception:
            msg = r.text
        raise HTTPError(f"{e} :: {msg}") from None
    return r.json()

def _drop_last_if_open_bars(bars: list) -> list:
    return bars[:-1] if len(bars) > 0 else bars

def _tf_minutes(tf: str) -> int:
    if tf.endswith("Min"):
        return int(tf.replace("Min", ""))
    if tf.endswith("Hour"):
        return int(tf.replace("Hour", "")) * 60
    if tf.endswith("Day"):
        return int(tf.replace("Day", "")) * 60 * 24
    return 15

def _iso(dt: datetime) -> str:
    # RFC3339 UTC with trailing Z (no +00:00Z)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _try_bars(symbol: str, tf: str, limit: int) -> list:
    # Ask far enough back using `start=`
    mins = _tf_minutes(tf)
    bars_needed = max(limit + 50, 600)
    minutes_back = bars_needed * mins
    from_when = _iso(now_utc() - timedelta(minutes=minutes_back))

    variants = []
    if DEFAULT_EXCH:
        variants.append({
            "symbols": f"{symbol}:{DEFAULT_EXCH}",
            "timeframe": tf,
            "start": from_when,
            "limit": min(10000, bars_needed + 200),
        })
    variants.append({
        "symbols": symbol,
        "timeframe": tf,
        "start": from_when,
        "limit": min(10000, bars_needed + 200),
    })

    last_error = None
    for params in variants:
        try:
            j = _alp_get("/v1beta3/crypto/us/bars", params)
            bars_map = j.get("bars", {})
            bars = (
                bars_map.get(symbol)
                or (bars_map.get(f"{symbol}:{DEFAULT_EXCH}") if DEFAULT_EXCH else None)
                or (bars_map.get(next(iter(bars_map))) if bars_map else None)
            )
            if bars:
                return bars
        except HTTPError as e:
            last_error = e
            continue

    if last_error:
        raise last_error
    return []

# One-time debug flag
__printed_counts = set()

def fetch_htf_closes(symbol: str, htf: str, limit: int) -> List[float]:
    tf = _map_tf_alp(htf)
    bars = _try_bars(symbol, tf, limit)
    bars = _drop_last_if_open_bars(bars)
    k = f"HTF:{symbol}:{tf}"
    if k not in __printed_counts:
        log_line(f"[DEBUG] got {len(bars)} HTF bars for {symbol} @ {tf}")
        __printed_counts.add(k)
    return [float(b["c"]) for b in bars]

def fetch_ltf_ohlc(symbol: str, ltf: str, limit: int) -> Tuple[List[float], List[float], List[float]]:
    tf = _map_tf_alp(ltf)
    bars = _try_bars(symbol, tf, limit)
    bars = _drop_last_if_open_bars(bars)
    k = f"LTF:{symbol}:{tf}"
    if k not in __printed_counts:
        log_line(f"[DEBUG] got {len(bars)} LTF bars for {symbol} @ {tf}")
        __printed_counts.add(k)
    highs  = [float(b["h"]) for b in bars]
    lows   = [float(b["l"]) for b in bars]
    closes = [float(b["c"]) for b in bars]
    return highs, lows, closes

def fetch_last_price(symbol: str) -> float:
    try:
        if DEFAULT_EXCH:
            j = _alp_get("/v1beta3/crypto/us/trades/latest", {"symbols": f"{symbol}:{DEFAULT_EXCH}"})
            t = j.get("trades", {}).get(f"{symbol}:{DEFAULT_EXCH}")
            if t and "p" in t:
                return float(t["p"])
        j = _alp_get("/v1beta3/crypto/us/trades/latest", {"symbols": symbol})
        t = j.get("trades", {}).get(symbol)
        if t and "p" in t:
            return float(t["p"])
    except Exception:
        pass
    return float("nan")
# ---------------------------------------------------------------------------





def size_for_risk(cash: float, last: float, stop: float, equity: float, allocated_now: float) -> float:
    if last is None or stop is None: return 0.0
    # risk sizing
    risk_dollars   = equity * RISK_PCT
    risk_per_unit  = max(1e-9, last - stop)
    qty_risk_based = risk_dollars / risk_per_unit
    # per-coin cap
    if PER_COIN_CAP >= 1.0:
        max_qty_by_coin_cap = PER_COIN_CAP / max(1e-9, last)     # dollars cap -> qty
    else:
        max_qty_by_coin_cap = (cash * PER_COIN_CAP) / max(1e-9, last)
    # portfolio cap
    max_port_notional = equity * PORTFOLIO_CAP
    room_left = max(0.0, max_port_notional - allocated_now)
    max_qty_by_port = room_left / max(1e-9, last)
    # cash cap
    max_qty_by_cash = cash / max(1e-9, last)
    qty = min(qty_risk_based, max_qty_by_coin_cap, max_qty_by_port, max_qty_by_cash)
    return max(0.0, qty)

def market_buy(st: Dict[str, Any], symbol: str, qty: float, price: float):
    if qty <= 0: return
    cost = qty * price
    if cost > st["cash"]:
        qty = st["cash"] / max(1e-9, price)
        cost = qty * price
    pos = get_pos(st, symbol)
    new_qty = pos["qty"] + qty
    if new_qty <= 0: return
    if pos["qty"] <= 0:
        pos["avg_entry"] = price
    else:
        pos["avg_entry"] = ((pos["avg_entry"] * pos["qty"]) + cost) / new_qty
    pos["qty"] = new_qty
    st["cash"] -= cost

def market_sell(st: Dict[str, Any], symbol: str, qty: float, price: float):
    if qty <= 0: return
    pos = get_pos(st, symbol)
    sell_qty = min(qty, pos["qty"])
    if sell_qty <= 0: return
    proceeds = sell_qty * price
    pos["qty"] -= sell_qty
    st["cash"] += proceeds
    if pos["qty"] == 0:
        pos["avg_entry"] = None
        pos["stop"] = None
        pos["tp1"] = None
        pos["tp2"] = None

def main():
    STARTED_AT = now_utc()
    state = load_state()
    global SYMBOLS
    SYMBOLS = [s for s in SYMBOLS if probe_symbol(s)]
    log_line(f"Active universe: {SYMBOLS}")
    ensure_trades_csv()
    init_daily_controls(state)

    while True:
        init_daily_controls(state)
        prices_now: Dict[str, float] = {}

        for sym in SYMBOLS:
            try:
                sym_state = get_sym_state(state, sym)
                pos = get_pos(state, sym)

                # Fetch data (closed bars)
                htf_closes = fetch_htf_closes(sym, HTF_RAW, max(WARMUP_HTF, 220))
                if len(htf_closes) < max(200 + HTF_SLOPE_N, WARMUP_HTF):
                    log_line(f"not enough TF bars for {sym}")
                    sym_state["warm_htf_ok"] = False
                    continue
                sym_state["warm_htf_ok"] = True

                ltf_highs, ltf_lows, ltf_closes = fetch_ltf_ohlc(sym, TF_RAW, max(WARMUP_LTF, DONCHIAN + ATR_LEN + 10))
                if len(ltf_closes) < WARMUP_LTF:
                    log_line(f"not enough TF bars for {sym}")
                    sym_state["warm_ltf_ok"] = False
                    continue
                sym_state["warm_ltf_ok"] = True

                last_close = ltf_closes[-1]
                last_tick  = fetch_last_price(sym)
                last = last_close if math.isnan(last_tick) else last_tick
                prices_now[sym] = last

                # HTF gate
                sma50  = sma(htf_closes, 50)
                sma200 = sma(htf_closes, 200)
                sma200_prev = sum(htf_closes[-200-HTF_SLOPE_N:-HTF_SLOPE_N]) / 200.0
                slope200    = (sma200 - sma200_prev) / float(HTF_SLOPE_N)
                last_c_htf  = htf_closes[-1]

                trend_ok = (sma50 > sma200)
                if TREND_MODE == "loose":
                    trend_ok = trend_ok or (last_c_htf > sma200 and slope200 > 0)
                elif TREND_MODE == "override":
                    dcH_200 = max(htf_closes[-200:])
                    breakout_override = (last_c_htf >= dcH_200)
                    trend_ok = trend_ok or (last_c_htf > sma200 and slope200 > 0) or breakout_override

                log_line(f"HTF trend {sym}: SMA50={round(sma50,2)}, SMA200={round(sma200,2)}")
                if not trend_ok and pos["qty"] <= 0:
                    log_line(f"{sym} HTF trend not OK; skip entry.")
                    sym_state["prev_price"] = last
                    continue

                # LTF indicators
                dcH, dcL = donchian(ltf_closes, DONCHIAN)
                A = atr(ltf_highs, ltf_lows, ltf_closes, ATR_LEN)
                if math.isnan(dcH) or math.isnan(A) or A <= 0:
                    log_line(f"{sym} indicators not ready; skip.")
                    sym_state["prev_price"] = last
                    continue

                # Targets
                if pos["qty"] > 0 and pos["avg_entry"]:
                    entry = pos["avg_entry"]
                    stop  = entry - STOP_ATR * A
                    tp1   = entry + (STOP_ATR * A)
                    tp2   = entry + (TP_MULT  * A)
                    pos["stop"], pos["tp1"], pos["tp2"] = stop, tp1, tp2
                else:
                    entry = dcH
                    stop  = entry - STOP_ATR * A
                    tp1   = entry + (STOP_ATR * A)
                    tp2   = entry + (TP_MULT  * A)

                # Startup mute
                if (now_utc() - STARTED_AT).total_seconds() < MUTE_SECS:
                    log_line(f"{sym} muted (startup)")
                    sym_state["prev_price"] = last
                    continue

                # Daily DD halt (entries only)
                if DAILY_DD_LIMIT <= 0.99:
                    start_eq = state.get("daily", {}).get("start_equity", state["equity"])
                    cur_dd = (start_eq - state["equity"]) / max(1e-9, start_eq)
                    if cur_dd >= DAILY_DD_LIMIT:
                        state["daily"]["halt"] = True
                if state.get("daily", {}).get("halt", False):
                    log_line("Daily DD limit reached; entries halted.")
                else:
                    # Entry (Donchian breakout)
                    if pos["qty"] <= 0 and trend_ok:
                        if crossed_up(sym_state["prev_price"], last, dcH) and can_fire(sym_state):
                            allocated = total_notional(state, prices_now)
                            qty = size_for_risk(state["cash"], dcH, stop, state["equity"], allocated)
                            if qty > 0:
                                market_buy(state, sym, qty, dcH)
                                pos = get_pos(state, sym)
                                pos["stop"], pos["tp1"], pos["tp2"] = stop, tp1, tp2
                                sym_state["tp1_hit"] = False
                                set_fired(sym_state)
                                write_trade(sym, "ENTRY", last, pos["avg_entry"], pos["stop"], pos["tp1"], pos["tp2"], qty, f"mode={TREND_MODE}")
                                log_line(f"{sym} ENTRY qty={qty:.6f} at={dcH:.6f}")

                # Exits
                if pos["qty"] > 0 and pos["avg_entry"]:
                    entry = pos["avg_entry"]; stop = pos["stop"]; tp1 = pos["tp1"]; tp2 = pos["tp2"]

                    if not sym_state.get("tp1_hit", False) and crossed_up(sym_state["prev_price"], last, tp1) and can_fire(sym_state):
                        sell_qty = pos["qty"] * 0.5
                        market_sell(state, sym, sell_qty, tp1)
                        sym_state["tp1_hit"] = True
                        set_fired(sym_state)
                        write_trade(sym, "TP1", last, entry, stop, tp1, tp2, sell_qty, "tp1")
                        log_line(f"{sym} TP1 sell qty={sell_qty:.6f} at={tp1:.6f}")

                    if pos["qty"] > 0 and crossed_up(sym_state["prev_price"], last, tp2) and can_fire(sym_state):
                        sell_qty = pos["qty"]
                        market_sell(state, sym, sell_qty, tp2)
                        set_fired(sym_state)
                        write_trade(sym, "EXIT", last, entry, stop, tp1, tp2, sell_qty, "tp2")
                        log_line(f"{sym} EXIT (TP2) qty={sell_qty:.6f} at={tp2:.6f}")

                    if pos["qty"] > 0 and crossed_down(sym_state["prev_price"], last, stop) and can_fire(sym_state):
                        sell_qty = pos["qty"]
                        market_sell(state, sym, sell_qty, stop)
                        set_fired(sym_state)
                        write_trade(sym, "EXIT", last, entry, stop, tp1, tp2, sell_qty, "stop")
                        log_line(f"{sym} EXIT (STOP) qty={sell_qty:.6f} at={stop:.6f}")

                # persist prev
                sym_state["prev_price"] = last

            except Exception as e:
                log_line(f"{sym} ERROR: {repr(e)}")

        # portfolio heartbeat
        try:
            mark_to_market(state, prices_now)
            save_state(state)
            log_line(f"Heartbeat: equity=${state['equity']:.2f}, cash=${state['cash']:.2f}")
        except Exception as e:
            log_line(f"MARK_TO_MARKET ERROR: {repr(e)}")

        time.sleep(LOOP_SECS)

if __name__ == "__main__":
    main()
