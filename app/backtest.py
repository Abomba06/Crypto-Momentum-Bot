import backtrader as bt, yfinance as yf, pandas as pd
from app.strategy import SmaCross

def run():
    df = yf.download("SPY", start="2018-01-01", auto_adjust=True, progress=False)

    # 🔧 Flatten MultiIndex columns (yfinance sometimes returns tuples)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    # 🔧 Ensure Backtrader-friendly column names
    if "Adj Close" in df.columns and "Close" not in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})

    data = bt.feeds.PandasData(dataname=df)  # uses Open/High/Low/Close/Volume
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100_000)
    cerebro.adddata(data)
    cerebro.addstrategy(SmaCross, fast=50, slow=200, risk=0.02)
    cerebro.run()
    print("Final equity:", round(cerebro.broker.getvalue(), 2))

if __name__ == "__main__":
    run()
