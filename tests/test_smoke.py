import unittest
from datetime import timedelta
from pathlib import Path

from app import backtest, crypto_trader


TEST_TMP_ROOT = Path("tests/.tmp")


def make_config(case_name: str) -> crypto_trader.BotConfig:
    root = TEST_TMP_ROOT / case_name
    root.mkdir(parents=True, exist_ok=True)
    return crypto_trader.BotConfig(
        symbols=["BTC/USD"],
        ltf_raw="5min",
        htf_raw="4hour",
        donchian=20,
        atr_len=14,
        stop_atr=1.8,
        tp1_atr=1.5,
        tp2_atr=3.0,
        trail_atr=2.0,
        breakout_buffer_bps=8.0,
        risk_pct=0.02,
        per_coin_cap=600.0,
        portfolio_cap=0.5,
        trend_mode="loose",
        htf_fast=50,
        htf_slow=200,
        htf_slope_n=10,
        ltf_fast_ema=21,
        ltf_slow_ema=55,
        rsi_len=14,
        rsi_entry_min=56.0,
        min_atr_pct=0.0035,
        volume_window=20,
        min_volume_ratio=1.1,
        warmup_ltf=70,
        warmup_htf=220,
        mute_secs=90,
        cooldown_secs=180,
        loop_secs=15,
        logs_dir=root,
        run_log=root / "run.log",
        trades_csv=root / "trades.csv",
        state_path=root / "state.json",
        initial_cash=100000.0,
        daily_dd_limit=0.05,
        alpaca_data_url="https://data.alpaca.markets",
        apca_key_id="key",
        apca_secret_key="secret",
        trading_base_url="https://paper-api.alpaca.markets",
        default_exchange="",
        request_timeout_secs=20,
    )


class StrategySmokeTests(unittest.TestCase):
    def test_backtest_entrypoint_exists(self):
        self.assertTrue(callable(backtest.run))

    def test_size_for_risk_respects_caps(self):
        config = make_config("size_for_risk")
        qty = crypto_trader.size_for_risk(
            config=config,
            cash=100000.0,
            last=200.0,
            stop=190.0,
            equity=100000.0,
            allocated_now=0.0,
        )
        self.assertGreater(qty, 0.0)
        self.assertLessEqual(qty * 200.0, 600.0 + 1e-9)

    def test_build_entry_signal_requires_breakout_conditions(self):
        config = make_config("entry_signal")
        closes = [100 + i * 0.2 for i in range(80)]
        highs = [c + 0.8 for c in closes]
        lows = [c - 0.7 for c in closes]
        volumes = [100.0] * 79 + [160.0]
        signal = crypto_trader.build_entry_signal(config, closes, highs, lows, volumes)
        self.assertIsNotNone(signal)
        self.assertGreater(signal.tp2, signal.tp1)
        self.assertGreater(signal.tp1, signal.breakout_level)

    def test_can_fire_uses_cooldown(self):
        config = make_config("cooldown")
        sym_state = {"last_fire_ts": (crypto_trader.now_utc() - timedelta(seconds=60)).timestamp()}
        self.assertFalse(crypto_trader.can_fire(config, sym_state))


if __name__ == "__main__":
    unittest.main()
