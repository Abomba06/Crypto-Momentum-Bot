# app/strategy.py
import pandas as pd
import numpy as np

def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/n, adjust=False).mean()
    roll_dn = pd.Series(down, index=series.index).ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / (roll_dn + 1e-12)
    return 100 - (100 / (1 + rs))

def donchian(df: pd.DataFrame, n: int) -> pd.DataFrame:
    hi = df["high"].rolling(n).max()
    lo = df["low"].rolling(n).min()
    mid = (hi + lo) / 2
    out = df.copy()
    out["dc_hi"] = hi
    out["dc_lo"] = lo
    out["dc_mid"] = mid
    return out

def atr(df: pd.DataFrame, n: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([
        (h - l),
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def make_indicators(df: pd.DataFrame, fast=8, slow=21, dc=10, atr_len=10):
    d = donchian(df, dc)
    d["sma_f"] = sma(d["close"], fast)
    d["sma_s"] = sma(d["close"], slow)
    d["rsi"]   = rsi(d["close"], 14)
    d["atr"]   = atr(d, atr_len)
    return d

def signal_row(row_prev, row_now) -> str:
    """Aggressive entry: (SMA cross up OR close>dc_hi*0.999) AND RSI>52.
       Exit: SMA cross down OR close<dc_mid."""
    # BUY conditions
    cross_up = row_prev.sma_f <= row_prev.sma_s and row_now.sma_f > row_now.sma_s
    breakout = row_now.close > (row_now.dc_hi * 0.999)
    momentum = row_now.rsi > 52
    if (cross_up or breakout) and momentum:
        return "BUY"

    # SELL conditions
    cross_dn = row_prev.sma_f >= row_prev.sma_s and row_now.sma_f < row_now.sma_s
    lose_trend = row_now.close < row_now.dc_mid
    if cross_dn or lose_trend:
        return "SELL"

    return "HOLD"

def compute_stops_tp(price: float, atr_val: float, stop_atr=1.6, tp_mult=1.6):
    stop = price - atr_val * stop_atr
    tp   = price + atr_val * tp_mult
    return stop, tp
