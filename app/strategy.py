import backtrader as bt
import numpy as np
import pandas as pd

try:
    from sklearn.mixture import GaussianMixture  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    GaussianMixture = None


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


def adaptive_regime_detector(closes: list[float], lookback: int = 252) -> dict:
    if len(closes) < max(30, lookback // 4):
        return {"name": "insufficient", "trend_score": 0.0, "vol_score": 0.0, "risk_multiplier": 1.0}
    series = pd.Series(closes[-lookback:], dtype="float64")
    returns = series.pct_change().dropna()
    if returns.empty:
        return {"name": "insufficient", "trend_score": 0.0, "vol_score": 0.0, "risk_multiplier": 1.0}
    realized_vol = float(returns.rolling(20).std().iloc[-1] * np.sqrt(252)) if len(returns) >= 20 else float(returns.std() * np.sqrt(252))
    momentum = float((series.iloc[-1] / max(series.iloc[-20], 1e-9)) - 1.0) if len(series) > 20 else 0.0
    trend_score = momentum * 10.0
    vol_score = realized_vol

    if GaussianMixture is not None and len(returns) >= 60:
        feats = pd.DataFrame(
            {
                "ret": returns,
                "vol": returns.rolling(10).std().fillna(returns.std()),
                "mom": returns.rolling(10).mean().fillna(0.0),
            }
        ).dropna()
        if len(feats) >= 30:
            model = GaussianMixture(n_components=3, covariance_type="full", random_state=7)
            labels = model.fit_predict(feats)
            latest = int(labels[-1])
            cluster = feats.iloc[np.where(labels == latest)[0]]
            cluster_mom = float(cluster["mom"].mean())
            cluster_vol = float(cluster["vol"].mean())
            if cluster_mom > 0 and cluster_vol < feats["vol"].quantile(0.6):
                return {"name": "bull_trend", "trend_score": trend_score, "vol_score": vol_score, "risk_multiplier": 1.12}
            if cluster_mom < 0 and cluster_vol > feats["vol"].quantile(0.6):
                return {"name": "bear_shock", "trend_score": trend_score, "vol_score": vol_score, "risk_multiplier": 0.65}
            return {"name": "range", "trend_score": trend_score, "vol_score": vol_score, "risk_multiplier": 0.88}

    if momentum > 0.03 and realized_vol < 0.55:
        return {"name": "bull_trend", "trend_score": trend_score, "vol_score": vol_score, "risk_multiplier": 1.10}
    if momentum < -0.03 and realized_vol > 0.45:
        return {"name": "bear_shock", "trend_score": trend_score, "vol_score": vol_score, "risk_multiplier": 0.70}
    if realized_vol < 0.30:
        return {"name": "range", "trend_score": trend_score, "vol_score": vol_score, "risk_multiplier": 0.92}
    return {"name": "transition", "trend_score": trend_score, "vol_score": vol_score, "risk_multiplier": 0.84}
