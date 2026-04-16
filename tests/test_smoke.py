import unittest
from datetime import timedelta
from pathlib import Path

import pandas as pd

from app import backtest, crypto_trader
from app import sentiment


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
        sentiment_enabled=False,
        sentiment_mode="disabled",
        sentiment_sources=["news"],
        sentiment_lookback_hours=24,
        sentiment_min_items=3,
        sentiment_buy_threshold=0.15,
        sentiment_sell_threshold=-0.15,
        sentiment_cache_secs=300,
        sentiment_exit_on_bearish=True,
        sentiment_keyword_map={},
        sentiment_news_limit=8,
        sentiment_twitter_limit=8,
        sentiment_twitter_rss_url="https://nitter.net/search/rss?f=tweets&q={query}",
        max_breakout_atr_extension=0.8,
        max_entry_rsi=72.0,
        ema_slope_lookback=3,
        dashboard_symbols=10,
        max_entries_per_loop=2,
        max_portfolio_heat=0.06,
        max_sector_exposure=0.35,
        weekly_dd_limit=0.10,
        max_consecutive_losses=4,
        pullback_ema_buffer_atr=0.35,
        shock_sentiment_threshold=-0.55,
        signals_csv=root / "signals.csv",
        research_report=root / "research_report.json",
        rs_lookback=20,
        min_relative_strength=-0.02,
        compression_window=12,
        compression_atr_ratio=0.85,
        reversal_lookback=10,
        min_execution_quality=0.55,
        news_momentum_min_acceleration=0.12,
        news_momentum_max_age_hours=6,
        news_momentum_min_recent_items=2,
        cross_asset_riskoff_penalty=0.72,
        cross_asset_alt_strength_bonus=0.08,
        failed_order_cooldown_secs=300,
        max_order_retries=1,
        thin_liquidity_size_penalty=0.86,
        reentry_window_secs=1800,
        max_reentries_per_symbol=1,
        strategy_kill_switch_enabled=True,
        max_theme_exposure=0.28,
        correlation_penalty_same_sector=0.88,
        correlation_penalty_same_theme=0.78,
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
        closes = [100 + ((i % 4) * 0.15) + i * 0.03 for i in range(80)]
        highs = [c + 0.8 for c in closes]
        lows = [c - 0.7 for c in closes]
        volumes = [100.0] * 79 + [160.0]
        signal = crypto_trader.build_entry_signal(config, closes, highs, lows, volumes)
        self.assertIsNotNone(signal)
        self.assertGreater(signal.tp2, signal.tp1)
        self.assertGreater(signal.tp1, signal.breakout_level)
        self.assertLessEqual(signal.confidence, 1.0)

    def test_can_fire_uses_cooldown(self):
        config = make_config("cooldown")
        sym_state = {"last_fire_ts": (crypto_trader.now_utc() - timedelta(seconds=60)).timestamp()}
        self.assertFalse(crypto_trader.can_fire(config, sym_state))

    def test_can_fire_respects_failed_order_block(self):
        config = make_config("order_block")
        sym_state = {"order_block_until_ts": (crypto_trader.now_utc() + timedelta(seconds=120)).timestamp()}
        self.assertFalse(crypto_trader.can_fire(config, sym_state))

    def test_build_sentiment_entry_signal_uses_atr_guardrails(self):
        config = make_config("sentiment_entry")
        closes = [100 + ((i % 4) * 0.15) + i * 0.03 for i in range(80)]
        highs = [c + 1.0 for c in closes]
        lows = [c - 0.9 for c in closes]
        volumes = [100.0] * 80
        signal = crypto_trader.build_sentiment_entry_signal(config, closes, highs, lows, volumes)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.source, "sentiment")
        self.assertLess(signal.stop, signal.breakout_level)

    def test_relative_strength_score_positive_when_outperforming(self):
        symbol = [100 + i * 1.0 for i in range(30)]
        bench = [100 + i * 0.5 for i in range(30)]
        self.assertGreater(crypto_trader.relative_strength_score(symbol, bench, 10), 0.0)

    def test_execution_quality_penalizes_worse_conditions(self):
        good = crypto_trader.estimate_execution_quality("BTC/USD", 0.01, 1.4)
        bad = crypto_trader.estimate_execution_quality("DOGE/USD", 0.06, 0.8)
        self.assertGreater(good, bad)

    def test_submit_order_with_safeguards_blocks_after_retries_fail(self):
        config = make_config("order_retry")
        sym_state = crypto_trader.get_sym_state(crypto_trader.default_state(100000.0), "DOGE/USD")

        class FailingBroker:
            def submit_market_buy(self, symbol, qty):
                raise RuntimeError("temporary venue issue")

        order, attempted_qty, error = crypto_trader.submit_order_with_safeguards(
            config,
            FailingBroker(),
            sym_state,
            "DOGE/USD",
            "buy",
            10.0,
        )
        self.assertIsNone(order)
        self.assertLess(attempted_qty, 10.0)
        self.assertIn("temporary venue issue", error)
        self.assertGreater(sym_state["order_fail_count"], 0)
        self.assertIsNotNone(sym_state["order_block_until_ts"])

    def test_eligible_reentry_after_stop_for_same_source(self):
        config = make_config("reentry")
        sym_state = crypto_trader.get_sym_state(crypto_trader.default_state(100000.0), "BTC/USD")
        sym_state["last_exit_reason"] = "stop"
        sym_state["last_exit_ts"] = crypto_trader.now_utc().timestamp()
        sym_state["last_entry_source_closed"] = "technical"
        sym_state["reentry_count"] = 1
        signal = crypto_trader.EntrySignal(
            breakout_level=100.0,
            stop=95.0,
            tp1=103.0,
            tp2=106.0,
            trail_anchor=96.0,
            atr_value=2.0,
            volume_ratio=1.2,
            rsi_value=60.0,
            source="technical",
            confidence=0.9,
            atr_pct=0.02,
            ema_spread_pct=0.01,
            reason="breakout",
        )
        self.assertTrue(crypto_trader.eligible_reentry(config, sym_state, signal))

    def test_compute_cross_asset_context_detects_risk_on(self):
        config = make_config("cross_asset")
        btc = [100 + i * 0.18 for i in range(config.warmup_htf + 5)]
        eth = [100 + i * 0.46 for i in range(config.warmup_htf + 5)]
        context = crypto_trader.compute_cross_asset_context(config, {"BTC/USD": btc, "ETH/USD": eth})
        self.assertEqual(context.regime, "risk-on")
        self.assertGreater(context.alts_multiplier, 1.0)

    def test_theme_exposure_groups_symbols_by_narrative(self):
        prices_now = {"SOL/USD": 100.0, "AVAX/USD": 50.0}
        positions = {
            "SOL/USD": crypto_trader.BrokerPosition("SOL/USD", 1.0, 1.0, 95.0, 100.0, 100.0),
            "AVAX/USD": crypto_trader.BrokerPosition("AVAX/USD", 2.0, 2.0, 48.0, 50.0, 100.0),
        }
        exposure = crypto_trader.theme_exposure(positions, prices_now, 1000.0)
        self.assertIn("high-beta-layer1", exposure)
        self.assertAlmostEqual(exposure["high-beta-layer1"], 0.2, places=6)

    def test_failed_breakdown_signal_can_trigger(self):
        config = make_config("reversal")
        closes = [100 + ((i % 3) * 0.015) for i in range(78)] + [99.2, 100.35]
        highs = [c + 0.9 for c in closes]
        lows = [c - 0.8 for c in closes]
        volumes = [100.0] * 79 + [160.0]
        signal = crypto_trader.build_failed_breakdown_signal(config, closes, highs, lows, volumes)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.source, "reversal")

    def test_build_news_momentum_signal_requires_fresh_bullish_catalyst(self):
        config = make_config("news_momentum")
        closes = [100 + ((i % 4) * 0.15) + i * 0.03 for i in range(80)]
        highs = [c + 1.0 for c in closes]
        lows = [c - 0.9 for c in closes]
        volumes = [100.0] * 80
        now = crypto_trader.now_utc()
        snapshot = sentiment.SentimentSnapshot(
            symbol="BTC/USD",
            score=0.42,
            label="bullish",
            sample_size=4,
            source_counts={"news": 4},
            items=[
                sentiment.SentimentItem("news", "reuters", "ETF approval sparks rally", now - timedelta(hours=1), 0.7, 1.2, 1.1, ["approval", "etf"]),
                sentiment.SentimentItem("news", "bloomberg", "Bitcoin gains after launch", now - timedelta(hours=2), 0.6, 1.1, 1.0, ["launch"]),
                sentiment.SentimentItem("news", "coindesk", "Adoption grows", now - timedelta(hours=8), 0.4, 0.9, 1.0, []),
                sentiment.SentimentItem("news", "coindesk", "More bullish flows", now - timedelta(hours=10), 0.3, 0.8, 1.0, []),
            ],
            top_headlines=["ETF approval sparks rally"],
            event_counts={"approval": 1, "etf": 1, "launch": 1},
            acceleration=0.24,
            updated_at=now,
        )
        signal = crypto_trader.build_news_momentum_signal(config, closes, highs, lows, volumes, snapshot)
        self.assertIsNotNone(signal)
        self.assertEqual(signal.source, "news_momentum")

    def test_confirm_mode_requires_positive_sentiment(self):
        config = make_config("confirm_mode")
        config = crypto_trader.BotConfig(**{**config.__dict__, "sentiment_enabled": True, "sentiment_mode": "confirm"})
        bullish = sentiment.SentimentSnapshot(
            symbol="BTC/USD",
            score=0.4,
            label="bullish",
            sample_size=5,
            source_counts={"news": 5},
            items=[],
            top_headlines=["ETF approval drives rally"],
            event_counts={"approval": 1},
            acceleration=0.2,
            updated_at=crypto_trader.now_utc(),
        )
        bearish = sentiment.SentimentSnapshot(
            symbol="BTC/USD",
            score=-0.4,
            label="bearish",
            sample_size=5,
            source_counts={"news": 5},
            items=[],
            top_headlines=["Hack triggers sell-off"],
            event_counts={"hack": 1},
            acceleration=-0.2,
            updated_at=crypto_trader.now_utc(),
        )
        self.assertTrue(crypto_trader.sentiment_allows_entry(config, bullish))
        self.assertFalse(crypto_trader.sentiment_allows_entry(config, bearish))

    def test_sentiment_scoring_moves_with_headline_tone(self):
        self.assertGreater(sentiment.score_text("Bitcoin rally surge after approval"), 0.0)
        self.assertLess(sentiment.score_text("Bitcoin crash after lawsuit and hack"), 0.0)

    def test_entry_signal_rejects_overextended_breakout(self):
        config = make_config("overextended")
        closes = [100 + i * 0.2 for i in range(79)] + [130.0]
        highs = [c + 1.0 for c in closes]
        lows = [c - 0.9 for c in closes]
        volumes = [100.0] * 79 + [200.0]
        signal = crypto_trader.build_entry_signal(config, closes, highs, lows, volumes)
        self.assertIsNone(signal)

    def test_detect_regime_flags_panic(self):
        snapshot = sentiment.SentimentSnapshot(
            symbol="BTC/USD",
            score=-0.8,
            label="bearish",
            sample_size=4,
            source_counts={"news": 4},
            items=[],
            top_headlines=["Exchange hack sparks panic"],
            event_counts={"hack": 1},
            acceleration=-0.4,
            updated_at=crypto_trader.now_utc(),
        )
        regime = crypto_trader.detect_regime(False, -1.5, 0.03, 1.4, snapshot)
        self.assertEqual(regime.name, "panic")
        self.assertFalse(regime.allow_breakout)

    def test_sentiment_client_parses_event_counts_and_headlines(self):
        class FakeResponse:
            def __init__(self, text):
                self.text = text

            def raise_for_status(self):
                pass

        class FakeSession:
            def get(self, url, timeout=None):
                return FakeResponse(
                    """<?xml version='1.0'?><rss><channel>
                    <item><title>Bitcoin ETF approval sparks rally - Reuters</title><description>Bitcoin rally after approval</description><pubDate>Wed, 15 Apr 2026 12:00:00 GMT</pubDate></item>
                    <item><title>Bitcoin hack fears fade as recovery starts - Bloomberg</title><description>Bitcoin recovery after hack scare</description><pubDate>Wed, 15 Apr 2026 11:00:00 GMT</pubDate></item>
                    <item><title>Bitcoin gains as adoption grows - CoinDesk</title><description>adoption growth and strong rebound</description><pubDate>Wed, 15 Apr 2026 10:00:00 GMT</pubDate></item>
                    </channel></rss>"""
                )

        client = sentiment.SentimentClient(
            FakeSession(),
            enabled=True,
            mode="confirm",
            sources=["news"],
            lookback_hours=48,
            min_items=3,
            bullish_threshold=0.15,
            bearish_threshold=-0.15,
            cache_secs=300,
        )
        snapshot = client.get_sentiment("BTC/USD", timeout=20)
        self.assertIsNotNone(snapshot)
        self.assertIn("approval", snapshot.event_counts)
        self.assertGreaterEqual(len(snapshot.top_headlines), 1)

    def test_build_research_report_aggregates_closed_trades(self):
        state = crypto_trader.default_state(100000.0)
        crypto_trader.update_trade_stats(state, "BTC/USD", "technical", "trend", 120.0, 3.2, {"theme": "store-of-value", "liquidity_tier": "liquid", "cross_asset_regime": "risk-on", "relative_strength_bucket": "strong", "execution_quality_bucket": "elite"})
        crypto_trader.update_trade_stats(state, "ETH/USD", "pullback", "trend", -50.0, -1.1, {"theme": "smart-contracts", "liquidity_tier": "liquid", "cross_asset_regime": "neutral", "relative_strength_bucket": "mixed", "execution_quality_bucket": "good"})
        report = crypto_trader.build_research_report(state)
        self.assertEqual(report["closed_trades"], 2)
        self.assertIn("technical", report["by_source"])
        self.assertIn("trend", report["by_regime"])
        self.assertIn("BTC/USD", report["by_symbol"])
        self.assertIn("expectancy", report)
        self.assertIn("health", report)
        self.assertIn("by_feature", report)
        self.assertIn("store-of-value", report["by_feature"]["theme"])

    def test_build_research_report_flags_degraded_health_when_expectancy_is_negative(self):
        state = crypto_trader.default_state(100000.0)
        for _ in range(6):
            crypto_trader.update_trade_stats(state, "BTC/USD", "technical", "trend", -40.0, -1.2)
        report = crypto_trader.build_research_report(state)
        self.assertEqual(report["health"]["status"], "degraded")
        self.assertIn("negative_expectancy", report["health"]["flags"])

    def test_build_runtime_alerts_surfaces_risk_conditions(self):
        config = make_config("alerts")
        state = crypto_trader.default_state(100000.0)
        state["daily"]["halt"] = True
        state["meta"]["consecutive_losses"] = config.max_consecutive_losses
        state["meta"]["no_candidate_loops"] = 4
        state["meta"]["cross_asset"] = {"regime": "risk-off"}
        report = {"health": {"status": "degraded"}}
        alerts = crypto_trader.build_runtime_alerts(config, state, report)
        self.assertTrue(any("Daily drawdown halt" in item for item in alerts))
        self.assertTrue(any("risk-off" in item for item in alerts))

    def test_build_live_artifact_summary_reads_recent_signal_and_trade_flow(self):
        config = make_config("live_artifacts")
        crypto_trader.write_signal_journal(config, "BTC/USD", "technical", "trend", 1.2, 0.2, 2.0, "accepted", "breakout")
        crypto_trader.write_signal_journal(config, "ETH/USD", "pullback", "trend", 1.0, 0.1, 1.5, "rejected", "theme_exposure")
        crypto_trader.write_trade(config, "BTC/USD", "ENTRY_SUBMITTED", 100.0, 100.0, 95.0, 103.0, 106.0, 1.0, "note")
        summary = crypto_trader.build_live_artifact_summary(config)
        self.assertEqual(summary["recent_entries"], 1)
        self.assertEqual(summary["recent_accepted_signals"], 1)
        self.assertEqual(summary["top_rejection_reasons"][0][0], "theme_exposure")

    def test_apply_live_source_tuning_disables_negative_source(self):
        state = crypto_trader.default_state(100000.0)
        report = {"by_source": {"technical": {"count": 5, "expectancy": -10.0}, "pullback": {"count": 5, "expectancy": 4.0}}}
        artifacts = {"accepted_by_setup": {"technical": 0, "pullback": 4}}
        crypto_trader.apply_live_source_tuning(state, report, artifacts)
        self.assertIn("technical", state["meta"]["disabled_sources"])

    def test_detect_behavior_shift_alerts_flags_major_changes(self):
        previous = {"candidate_count": 4, "cross_asset_regime": "risk-on", "top_setup": "technical", "strategy_halt": False}
        current = {"candidate_count": 0, "cross_asset_regime": "risk-off", "top_setup": "news_momentum", "strategy_halt": True}
        alerts = crypto_trader.detect_behavior_shift_alerts(previous, current)
        self.assertTrue(any("collapsed" in item for item in alerts))
        self.assertTrue(any("regime changed" in item for item in alerts))
        self.assertTrue(any("halt" in item for item in alerts))

    def test_strategy_kill_switch_sets_halt_when_health_is_degraded(self):
        config = make_config("strategy_halt")
        state = crypto_trader.default_state(100000.0)
        report = {"closed_trades": 9, "health": {"status": "degraded", "weak_sources": ["technical"]}}
        crypto_trader.update_strategy_health_halt(config, state, report)
        self.assertTrue(state["meta"]["strategy_halt"])
        self.assertIn("technical", state["meta"]["disabled_sources"])

    def test_execute_ranked_entries_respects_ranking_and_updates_state(self):
        config = make_config("execute_ranked")
        state = crypto_trader.default_state(100000.0)
        account = crypto_trader.AccountSnapshot(cash=100000.0, equity=100000.0)
        positions = {}
        prices_now = {"BTC/USD": 100.0}

        class FakeBroker:
            def __init__(self):
                self.orders = []

            def submit_market_buy(self, symbol, qty):
                self.orders.append((symbol, qty))
                return type("Order", (), {"id": f"{symbol}-1"})()

        btc_snapshot = sentiment.SentimentSnapshot(
            symbol="BTC/USD",
            score=0.4,
            label="bullish",
            sample_size=4,
            source_counts={"news": 4},
            items=[],
            top_headlines=["ETF approval"],
            event_counts={"approval": 1},
            acceleration=0.2,
            updated_at=crypto_trader.now_utc(),
        )
        candidate = crypto_trader.CandidateSetup(
            symbol="BTC/USD",
            signal=crypto_trader.EntrySignal(
                breakout_level=100.0,
                stop=95.0,
                tp1=103.0,
                tp2=106.0,
                trail_anchor=96.0,
                atr_value=2.0,
                volume_ratio=1.2,
                rsi_value=60.0,
                source="technical",
                confidence=0.9,
                atr_pct=0.02,
                ema_spread_pct=0.01,
                reason="breakout",
            ),
            regime=crypto_trader.RegimeSnapshot("trend", 1.0, True, True, True, 1.0),
            sentiment=btc_snapshot,
            trend_score=2.0,
            sector="majors",
            score=1.6,
            last_price=100.0,
            execution_quality=0.9,
            relative_strength=0.05,
            cross_asset_multiplier=1.04,
            liquidity_tier="liquid",
            theme="store-of-value",
        )
        crypto_trader.get_sym_state(state, "BTC/USD")["snapshot"] = {"symbol": "BTC/USD", "last_action": "watch"}
        broker = FakeBroker()
        crypto_trader.execute_ranked_entries(config, broker, state, account, positions, prices_now, [candidate], set())
        self.assertEqual(len(broker.orders), 1)
        self.assertEqual(state["sym"]["BTC/USD"]["entry_source"], "technical")

    def test_reconcile_positions_clears_stale_state_when_broker_flat(self):
        config = make_config("reconcile_flat")
        state = crypto_trader.default_state(100000.0)
        sym_state = crypto_trader.get_sym_state(state, "BTC/USD")
        sym_state["last_entry_price"] = 100.0
        sym_state["high_water"] = 105.0
        sym_state["entry_source"] = "technical"
        result = crypto_trader.reconcile_positions_with_state(config, state, ["BTC/USD"], {})
        self.assertEqual(result["cleared"], 1)
        self.assertIsNone(sym_state["entry_source"])

    def test_reconcile_positions_restores_missing_state_from_live_position(self):
        config = make_config("reconcile_live")
        state = crypto_trader.default_state(100000.0)
        position = crypto_trader.BrokerPosition(
            symbol="BTC/USD",
            qty=1.5,
            qty_available=1.5,
            avg_entry_price=101.0,
            current_price=103.0,
            market_value=154.5,
        )
        result = crypto_trader.reconcile_positions_with_state(config, state, ["BTC/USD"], {"BTC/USD": position})
        self.assertEqual(result["restored"], 1)
        self.assertEqual(state["sym"]["BTC/USD"]["last_entry_price"], 101.0)

    def test_backtest_simulation_returns_metrics(self):
        idx = pd.date_range("2024-01-01", periods=320, freq="D")
        closes = [100 + ((i % 5) * 0.12) + i * 0.08 for i in range(len(idx))]
        df = pd.DataFrame(
            {
                "open": closes,
                "high": [c + 1.0 for c in closes],
                "low": [c - 0.9 for c in closes],
                "close": closes,
                "volume": [1000 + (i % 7) * 50 for i in range(len(idx))],
            },
            index=idx,
        )
        result = backtest.simulate_strategy(df, backtest.BacktestConfig())
        self.assertIn("metrics", result)
        self.assertIn("ending_equity", result["metrics"])
        self.assertGreater(len(result["equity_curve"]), 0)

    def test_walk_forward_validate_returns_summary(self):
        idx = pd.date_range("2023-01-01", periods=500, freq="D")
        closes = [100 + ((i % 6) * 0.09) + i * 0.05 for i in range(len(idx))]
        df = pd.DataFrame(
            {
                "open": closes,
                "high": [c + 1.0 for c in closes],
                "low": [c - 0.9 for c in closes],
                "close": closes,
                "volume": [1200 + (i % 9) * 40 for i in range(len(idx))],
            },
            index=idx,
        )
        result = backtest.walk_forward_validate(df, backtest.BacktestConfig(train_bars=200, test_bars=100))
        self.assertIn("summary", result)
        self.assertGreaterEqual(len(result["windows"]), 1)
        self.assertIn("stability_score", result["summary"])

    def test_portfolio_backtest_returns_multi_symbol_metrics(self):
        idx = pd.date_range("2024-01-01", periods=320, freq="D")
        df_a = pd.DataFrame(
            {
                "open": [100 + ((i % 5) * 0.10) + i * 0.06 for i in range(len(idx))],
                "high": [101 + ((i % 5) * 0.10) + i * 0.06 for i in range(len(idx))],
                "low": [99 + ((i % 5) * 0.10) + i * 0.06 for i in range(len(idx))],
                "close": [100 + ((i % 5) * 0.10) + i * 0.06 for i in range(len(idx))],
                "volume": [1000 + (i % 7) * 30 for i in range(len(idx))],
            },
            index=idx,
        )
        df_b = pd.DataFrame(
            {
                "open": [80 + ((i % 4) * 0.12) + i * 0.05 for i in range(len(idx))],
                "high": [81 + ((i % 4) * 0.12) + i * 0.05 for i in range(len(idx))],
                "low": [79 + ((i % 4) * 0.12) + i * 0.05 for i in range(len(idx))],
                "close": [80 + ((i % 4) * 0.12) + i * 0.05 for i in range(len(idx))],
                "volume": [900 + (i % 5) * 35 for i in range(len(idx))],
            },
            index=idx,
        )
        result = backtest.simulate_portfolio_strategy(
            {"AAA": df_a, "BBB": df_b},
            backtest.BacktestConfig(symbols=["AAA", "BBB"], max_entries_per_bar=2),
        )
        self.assertIn("metrics", result)
        self.assertEqual(sorted(result["metrics"]["symbols"]), ["AAA", "BBB"])
        self.assertGreater(len(result["equity_curve"]), 0)

    def test_parameter_sweep_returns_ranked_results(self):
        idx = pd.date_range("2024-01-01", periods=320, freq="D")
        closes = [100 + ((i % 5) * 0.10) + i * 0.06 for i in range(len(idx))]
        df = pd.DataFrame(
            {
                "open": closes,
                "high": [c + 1.0 for c in closes],
                "low": [c - 0.9 for c in closes],
                "close": closes,
                "volume": [1000 + (i % 7) * 30 for i in range(len(idx))],
            },
            index=idx,
        )
        result = backtest.parameter_sweep(
            df,
            backtest.BacktestConfig(),
            {"donchian": [16, 20], "rsi_entry_min": [54.0, 56.0]},
        )
        self.assertEqual(result["evaluated"], 4)
        self.assertIsNotNone(result["best"])
        self.assertGreaterEqual(len(result["top"]), 1)
        self.assertIn("walk_forward", result["best"])
        self.assertIn("robustness", result["best"])


if __name__ == "__main__":
    unittest.main()
