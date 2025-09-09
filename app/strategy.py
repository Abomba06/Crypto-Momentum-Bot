import math
import backtrader as bt
import numpy as np

class SmaCross(bt.Strategy):
    params = dict(fast=50, slow=200, risk=0.02)

    def __init__(self):
        self.sma_fast = bt.ind.SMA(period=self.p.fast)
        self.sma_slow = bt.ind.SMA(period=self.p.slow)
        self.cross = bt.ind.CrossOver(self.sma_fast, self.sma_slow)

    def next(self):
        price = float(self.data.close[0])

        # ⛔ skip until indicators have valid values (no NaNs)
        if (np.isnan(self.sma_fast[0]) or
            np.isnan(self.sma_slow[0]) or
            math.isnan(price)):
            return

        cash = self.broker.getcash()
        size = int(max(0, (cash * self.p.risk) / price))

        if not self.position and self.cross > 0 and size > 0:
            self.buy(size=size)
        elif self.position and self.cross < 0:
            self.close()
