import backtrader as bt
import pandas as pd
import yfinance as yf

from app.strategy import SmaCross


def load_price_history(symbol: str, start: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Adj Close": "close",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[required].dropna()


def run(symbol: str = "SPY", start: str = "2018-01-01", cash: float = 100_000) -> float:
    df = load_price_history(symbol, start)
    data = bt.feeds.PandasData(dataname=df)

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(cash)
    cerebro.adddata(data)
    cerebro.addstrategy(SmaCross, fast=50, slow=200, risk=0.02)
    cerebro.run()

    final_equity = round(cerebro.broker.getvalue(), 2)
    print(f"Final equity for {symbol}: {final_equity}")
    return final_equity


if __name__ == "__main__":
    run()
