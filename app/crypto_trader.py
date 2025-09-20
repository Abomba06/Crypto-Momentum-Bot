# crypto_trader_v2.py
# Minimal paper-trading loop with HTF trend gate (strict/loose/override),
# ATR stops, TP1/TP2, startup mute, cooldown, cross-detection, warmups,
# and CSV logging. Data fetch stubs provided—wire to your data source.

import os, time, json, math, csv, pathlib
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Tuple

# ----------------------- ENV / CONFIG ----------------------------------------
SYMBOLS     = [s.strip() for s in os.getenv("CRYPTO_SYMBOLS", "BTC/USD,ETH/USD,SOL/USD,XRP/USD,LINK/USD,AVAX/USD,DOGE/USD,LTC/USD,UNI/USD,ADA/USD,MATIC/USD").split(",") if s.strip()]
TF_RAW      = os.getenv("CRYPTO_TIMEFRAME", "5min").lower()     # LTF (entry/exit mgmt)
HTF_RAW     = os.getenv("HTF_TIMEFRAME", "day").lower()         # HTF (trend filter)

DONCHIAN    = int(os.getenv("DONCHIAN", "30"))                  # LTF Donchian breakout window
ATR_LEN     = int(os.getenv("ATR_LEN", "14"))
STOP_ATR    = float(os.getenv("STOP_ATR", "2.5"))
TP_MULT     = float(os.getenv("TP_MULT", "2.0"))                # TP2 multiple of ATR vs entry
TRAIL_ATR   = float(os.getenv("TRAIL_ATR", "0.0"))              # 0 = off

RISK_PCT       = float(os.getenv("RISK_PCT", "0.005"))          # 0.5% per trade
PER_COIN_CAP   = float(os.getenv("PER_COIN_CAP", "0.20"))       # max exposure per symbol (20%)

TREND_MODE     = os.getenv("TREND_MODE", "strict").lower()      # strict | loose | override
WARMUP_LTF     = int(os.getenv("WARMUP_LTF", str(max(DONCHIAN, ATR_LEN)+10)))
WARMUP_HTF     = int(os.getenv("WARMUP_HTF", "205"))            # need >= 200 + slope lookback
HTF_SLOPE_N    = int(os.getenv("HTF_SLOPE_LOOKBACK", "10"))

MUTE_SECS      = int(os.getenv("MUTE_SECS", "90"))              # boot mute to avoid latent triggers
COOLDOWN_SECS  = int(os.getenv("COOLDOWN_SECS", "10"))
LOOP_SECS      = int(os.getenv("LOOP_SECS", "15"))

DEBUG          = os.getenv("DEBUG", "1") == "1"

# ----------------------- FILES / FOLDERS -------------------------------------
BASE = pathlib.Path(".")
LOGS = BASE / "logs"
LOGS.mkdir(exist_ok=True)
RUN_LOG = LOGS / "run.log"
TRD_CSV = LOGS / "trades.csv"
STATE_F = BASE / "state.json"

# ----------------------- UTILITIES -------------------------------------------
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def hb(msg: str, **kw):
    if DEBUG:
        print(f"[{now_utc().isoformat()}] {msg} " + " ".join(f"{k}={v}" for k,v in kw.items()))

def log_line(s: str):
    with RUN_LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{now_utc().isoformat()}] {s}\n")

def ensure_trades_csv():
    header = ["ts","symbol","action","price_last","avg_entry","stop","tp1","tp2","qty","note"]
    new = not TRD_CSV.exists()
    with TRD_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)

def write_trade(symbol, action, price_last, avg_entry, stop, tp1, tp2, qty, note=""):
    ensure_trades_csv()
    with TRD_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([now_utc().isoformat(), symbol, action, f"{price_last:.6f}",
                    f"{(avg_entry or 0):.6f}", f"{(stop or 0):.6f}",
                    f"{(tp1 or 0):.6f}", f"{(tp2 or 0):.6f}", f"{qty:.8f}", note])

def crossed_up(prev: float, now: float, level: float) -> bool:
    return (prev is not None) and (prev < level <= now)

def crossed_down(prev: float, now: float, level: float) -> bool:
    return (prev is not None) and (prev > level >= now)

def sma(seq: List[float], n: int) -> float:
    if len(seq) < n: return float('nan')
    return sum(seq[-n:]) / float(n)

def donchian(seq: List[float], n: int) -> Tuple[float,float]:
    if len(seq) < n: return float('nan'), float('nan')
    window = seq[-n:]
    return max(window), min(window)

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

def parse_cash(v) -> float:
    try: return float(v)
    except: return 100000.0

# ----------------------- DATA FETCH (TODO: wire these) -----------------------
def fetch_htf_closes(symbol: str, htf: str, limit: int) -> List[float]:
    """
    TODO: Replace with your real data provider.
    Must return a list of CLOSED HTF candle closes (e.g., daily closes), oldest->newest.
    """
    raise NotImplementedError("fetch_htf_closes(symbol, htf, limit)")

def fetch_ltf_ohlc(symbol: str, ltf: str, limit: int) -> Tuple[List[float], List[float], List[float]]:
    """
    TODO: Replace with your real data provider.
    Must return (highs, lows, closes) lists for LTF candles, CLOSED candles only, oldest->newest.
    """
    raise NotImplementedError("fetch_ltf_ohlc(symbol, ltf, limit)")

def fetch_last_price(symbol: str) -> float:
    """
    Optional: fast tick/last price for cross detection between bars.
    If not available, fall back to last LTF close.
    """
    # You can implement via your feed; placeholder returns NaN to force fallback.
    return float('nan')

# ----------------------- STATE / BROKER --------------------------------------
DEFAULT_STATE = {
    "cash": 100000.0,
    "equity": 100000.0,
    "positions": {},         # symbol -> dict
    "sym": {}                # symbol -> misc per-symbol state
}

def load_state() -> Dict[str, Any]:
    if STATE_F.exists():
        try:
            return json.loads(STATE_F.read_text())
        except:
            pass
    return DEFAULT_STATE.copy()

def save_state(st: Dict[str, Any]):
    STATE_F.write_text(json.dumps(st, indent=2))

def get_sym_state(st: Dict[str, Any], sym: str) -> Dict[str, Any]:
    s = st["sym"].get(sym)
    if not s:
        s = {
            "prev_price": None,
            "last_fire_ts": None,
            "tp1_hit": False,
            "warm_ltf_ok": False,
            "warm_htf_ok": False,
        }
        st["sym"][sym] = s
    return s

def get_pos(st: Dict[str, Any], sym: str) -> Dict[str, Any]:
    p = st["positions"].get(sym)
    if not p:
        p = {
            "qty": 0.0,
            "avg_entry": None,
            "stop": None,
            "tp1": None,
            "tp2": None,
        }
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
    if last is None: return True
    return (ts - float(last)) >= COOLDOWN_SECS

def set_fired(sym_state: Dict[str, Any]):
    sym_state["last_fire_ts"] = now_utc().timestamp()

# ----------------------- POSITION SIZING -------------------------------------
def size_for_risk(cash: float, last: float, stop: float) -> float:
    if last is None or stop is None: return 0.0
    risk_dollars = cash * RISK_PCT
    risk_per_unit = max(1e-9, last - stop)  # long risk
    qty = risk_dollars / risk_per_unit
    # cap exposure per symbol
    max_qty_by_cap = (cash * PER_COIN_CAP) / max(1e-9, last)
    return max(0.0, min(qty, max_qty_by_cap))

# ----------------------- PAPER BROKER ----------------------------------------
def market_buy(st: Dict[str, Any], symbol: str, qty: float, price: float):
    if qty <= 0: return
    cost = qty * price
    if cost > st["cash"]:
        qty = st["cash"] / max(1e-9, price)
        cost = qty * price
    pos = get_pos(st, symbol)
    new_qty = pos["qty"] + qty
    if new_qty <= 0:
        return
    # average entry
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

# ----------------------- MAIN LOOP -------------------------------------------
def main():
    STARTED_AT = now_utc()
    state = load_state()
    ensure_trades_csv()

    while True:
        prices_now = {}
        for sym in SYMBOLS:
            try:
                sym_state = get_sym_state(state, sym)
                pos = get_pos(state, sym)

                # 1) Fetch CLOSED bars for HTF and LTF
                # Warm-ups
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
                last_tick = fetch_last_price(sym)
                if math.isnan(last_tick):
                    last = last_close
                else:
                    last = last_tick
                prices_now[sym] = last

                # 2) HTF trend gate ------------------------------------------
                sma50  = sma(htf_closes, 50)
                sma200 = sma(htf_closes, 200)
                sma200_prev = sum(htf_closes[-200-HTF_SLOPE_N:-HTF_SLOPE_N]) / 200.0
                slope200    = (sma200 - sma200_prev) / float(HTF_SLOPE_N)
                last_c_htf  = htf_closes[-1]

                trend_ok = (sma50 > sma200)
                if TREND_MODE == "loose":
                    trend_ok = trend_ok or (last_c_htf > sma200 and slope200 > 0)
                elif TREND_MODE == "override":
                    dcH_200, _ = max(htf_closes[-200:]), min(htf_closes[-200:])
                    breakout_override = (last_c_htf >= dcH_200)
                    trend_ok = trend_ok or (last_c_htf > sma200 and slope200 > 0) or breakout_override

                log_line(f"HTF trend {sym}: SMA50={round(sma50,2)}, SMA200={round(sma200,2)}")
                if not trend_ok and pos["qty"] <= 0:
                    log_line(f"{sym} HTF trend not OK; skip entry.")
                    continue

                # 3) LTF indicators ------------------------------------------
                dcH, dcL = donchian(ltf_closes, DONCHIAN)
                A = atr(ltf_highs, ltf_lows, ltf_closes, ATR_LEN)
                if math.isnan(dcH) or math.isnan(A) or A <= 0:
                    log_line(f"{sym} indicators not ready; skip.")
                    continue

                # 4) Calculate stops/targets for current or hypothetical pos --
                if pos["qty"] > 0 and pos["avg_entry"]:
                    entry = pos["avg_entry"]
                    stop  = entry - STOP_ATR * A
                    tp1   = entry + (STOP_ATR * A)          # take first profits ~1R
                    tp2   = entry + (TP_MULT  * A)          # runner target
                    pos["stop"], pos["tp1"], pos["tp2"] = stop, tp1, tp2
                else:
                    # hypothetical (for sizing)
                    entry = dcH
                    stop  = entry - STOP_ATR * A
                    tp1   = entry + (STOP_ATR * A)
                    tp2   = entry + (TP_MULT  * A)

                # 5) Startup mute --------------------------------------------
                if (now_utc() - STARTED_AT).total_seconds() < MUTE_SECS:
                    log_line(f"{sym} muted (startup)")
                    sym_state["prev_price"] = last
                    continue

                # 6) Entry logic (Donchian breakout long) --------------------
                did_trade = False
                if pos["qty"] <= 0 and trend_ok:
                    # Use cross to avoid startup latent triggers
                    if crossed_up(sym_state["prev_price"], last, dcH) and can_fire(sym_state):
                        # size
                        qty = size_for_risk(state["cash"], dcH, stop)
                        if qty > 0:
                            market_buy(state, sym, qty, dcH)
                            # lock levels from *current* A at fill
                            pos = get_pos(state, sym)  # refresh
                            pos["stop"] = stop
                            pos["tp1"]  = tp1
                            pos["tp2"]  = tp2
                            sym_state["tp1_hit"] = False
                            set_fired(sym_state)
                            write_trade(sym, "ENTRY", last, pos["avg_entry"], pos["stop"], pos["tp1"], pos["tp2"], qty, f"mode={TREND_MODE}")
                            did_trade = True
                            log_line(f"{sym} ENTRY qty={qty:.6f} at={dcH:.6f}")

                # 7) Exit / scale logic for open positions -------------------
                if pos["qty"] > 0 and pos["avg_entry"]:
                    entry = pos["avg_entry"]
                    stop  = pos["stop"]
                    tp1   = pos["tp1"]
                    tp2   = pos["tp2"]

                    # TP1 half-off (once)
                    if not sym_state.get("tp1_hit", False) and crossed_up(sym_state["prev_price"], last, tp1) and can_fire(sym_state):
                        sell_qty = pos["qty"] * 0.5
                        market_sell(state, sym, sell_qty, tp1)
                        sym_state["tp1_hit"] = True
                        set_fired(sym_state)
                        write_trade(sym, "TP1", last, entry, stop, tp1, tp2, sell_qty, "tp1")
                        log_line(f"{sym} TP1 sell qty={sell_qty:.6f} at={tp1:.6f}")
                        did_trade = True

                    # Final TP2
                    if pos["qty"] > 0 and crossed_up(sym_state["prev_price"], last, tp2) and can_fire(sym_state):
                        sell_qty = pos["qty"]
                        market_sell(state, sym, sell_qty, tp2)
                        set_fired(sym_state)
                        write_trade(sym, "EXIT", last, entry, stop, tp1, tp2, sell_qty, "tp2")
                        log_line(f"{sym} EXIT (TP2) qty={sell_qty:.6f} at={tp2:.6f}")
                        did_trade = True

                    # Stop-loss
                    if pos["qty"] > 0 and crossed_down(sym_state["prev_price"], last, stop) and can_fire(sym_state):
                        sell_qty = pos["qty"]
                        market_sell(state, sym, sell_qty, stop)
                        set_fired(sym_state)
                        write_trade(sym, "EXIT", last, entry, stop, tp1, tp2, sell_qty, "stop")
                        log_line(f"{sym} EXIT (STOP) qty={sell_qty:.6f} at={stop:.6f}")
                        did_trade = True

                # 8) Persist prev_price for cross detection ------------------
                sym_state["prev_price"] = last

            except NotImplementedError as e:
                # Data fetch not wired yet
                log_line(f"{sym} DATA_FETCH_NOT_IMPLEMENTED: {e}")
            except Exception as e:
                log_line(f"{sym} ERROR: {repr(e)}")

        # 9) Mark-to-market & heartbeat ---------------------------------------
        try:
            mark_to_market(state, prices_now)
            save_state(state)
            log_line(f"Heartbeat: equity=${state['equity']:.2f}, cash=${state['cash']:.2f}")
        except Exception as e:
            log_line(f"MARK_TO_MARKET ERROR: {repr(e)}")

        time.sleep(LOOP_SECS)

# ----------------------- ENTRY POINT -----------------------------------------
if __name__ == "__main__":
    main()
