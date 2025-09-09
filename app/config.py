import os
from dotenv import load_dotenv
load_dotenv()

ALPACA_KEY_ID     = os.getenv("ALPACA_KEY_ID")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

SYMBOL    = os.getenv("SYMBOL", "SPY")
FAST      = int(os.getenv("FAST", "50"))
SLOW      = int(os.getenv("SLOW", "200"))
RISK_PCT  = float(os.getenv("RISK_PCT", "0.02"))
TIMEFRAME = os.getenv("TIMEFRAME", "day")
