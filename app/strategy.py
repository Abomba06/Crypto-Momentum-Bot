import backtrader as bt
import numpy as np
import pandas as pd


def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()


def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1 / n, adjust=False).mean()
    roll_dn = pd.Series(down, index=series.index).ewm(alpha=1 / n, adjust=False).mean()
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
    tr = pd.concat(
        [
            (h - l),
            (h - c.shift()).abs(),
            (l - c.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean()


def make_indicators(df: pd.DataFrame, fast: int = 8, slow: int = 21, dc: int = 10, atr_len: int = 10) -> pd.DataFrame:
    data = donchian(df, dc)
    data["sma_f"] = sma(data["close"], fast)
    data["sma_s"] = sma(data["close"], slow)
    data["rsi"] = rsi(data["close"], 14)
    data["atr"] = atr(data, atr_len)
    return data


def signal_row(row_prev, row_now) -> str:
    cross_up = row_prev.sma_f <= row_prev.sma_s and row_now.sma_f > row_now.sma_s
    breakout = row_now.close > (row_now.dc_hi * 0.999)
    momentum = row_now.rsi > 52
    if (cross_up or breakout) and momentum:
        return "BUY"

    cross_down = row_prev.sma_f >= row_prev.sma_s and row_now.sma_f < row_now.sma_s
    lose_trend = row_now.close < row_now.dc_mid
    if cross_down or lose_trend:
        return "SELL"

    return "HOLD"


def compute_stops_tp(price: float, atr_val: float, stop_atr: float = 1.6, tp_mult: float = 1.6):
    stop = price - atr_val * stop_atr
    tp = price + atr_val * tp_mult
    return stop, tp


class SmaCross(bt.Strategy):
    params = (
        ("fast", 50),
        ("slow", 200),
        ("risk", 0.02),
    )

    def __init__(self):
        self.fast_sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.fast)
        self.slow_sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.slow)
        self.cross = bt.indicators.CrossOver(self.fast_sma, self.slow_sma)

    def next(self):
        if self.position.size == 0 and self.cross > 0:
            cash = self.broker.getcash()
            target_value = max(0.0, cash * self.p.risk)
            size = int(target_value / max(self.data.close[0], 1e-9))
            if size > 0:
                self.buy(size=size)
        elif self.position.size > 0 and self.cross < 0:
            self.close()
