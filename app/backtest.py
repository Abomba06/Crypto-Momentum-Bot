from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

from app import crypto_trader


@dataclass(frozen=True)
class BacktestConfig:
    symbol: str = "SPY"
    symbols: List[str] = field(default_factory=list)
    start: str = "2018-01-01"
    interval: str = "1d"
    cash: float = 100_000.0
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    train_bars: int = 252
    test_bars: int = 126
    risk_pct: float = 0.01
    per_trade_cap: float = 0.20
    portfolio_cap: float = 1.0
    donchian: int = 20
    atr_len: int = 14
    stop_atr: float = 1.8
    tp1_atr: float = 1.5
    tp2_atr: float = 3.0
    trail_atr: float = 2.0
    breakout_buffer_bps: float = 8.0
    htf_fast: int = 30
    htf_slow: int = 100
    htf_slope_n: int = 10
    ltf_fast_ema: int = 21
    ltf_slow_ema: int = 55
    rsi_len: int = 14
    rsi_entry_min: float = 56.0
    min_atr_pct: float = 0.0035
    volume_window: int = 20
    min_volume_ratio: float = 1.05
    max_breakout_atr_extension: float = 0.8
    max_entry_rsi: float = 72.0
    ema_slope_lookback: int = 3
    pullback_ema_buffer_atr: float = 0.35
    trend_mode: str = "loose"
    max_entries_per_bar: int = 2

    @property
    def warmup_ltf(self) -> int:
        return max(
            self.donchian + 5,
            self.atr_len + 5,
            self.ltf_slow_ema + 5,
            self.volume_window + 5,
            self.rsi_len + 5,
        )

    @property
    def warmup_htf(self) -> int:
        return max(self.htf_slow + self.htf_slope_n + 5, 220)


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("No data returned")
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


def load_price_history(symbol: str, start: str, interval: str = "1d") -> pd.DataFrame:
    df = yf.download(symbol, start=start, interval=interval, auto_adjust=True, progress=False)
    return normalize_ohlcv(df)


def load_price_histories(symbols: List[str], start: str, interval: str = "1d") -> Dict[str, pd.DataFrame]:
    return {symbol: load_price_history(symbol, start, interval) for symbol in symbols}


def as_runtime_config(config: BacktestConfig) -> Any:
    return SimpleNamespace(
        donchian=config.donchian,
        atr_len=config.atr_len,
        stop_atr=config.stop_atr,
        tp1_atr=config.tp1_atr,
        tp2_atr=config.tp2_atr,
        trail_atr=config.trail_atr,
        breakout_buffer_bps=config.breakout_buffer_bps,
        risk_pct=config.risk_pct,
        per_coin_cap=config.per_trade_cap,
        portfolio_cap=config.portfolio_cap,
        trend_mode=config.trend_mode,
        htf_fast=config.htf_fast,
        htf_slow=config.htf_slow,
        htf_slope_n=config.htf_slope_n,
        ltf_fast_ema=config.ltf_fast_ema,
        ltf_slow_ema=config.ltf_slow_ema,
        rsi_len=config.rsi_len,
        rsi_entry_min=config.rsi_entry_min,
        min_atr_pct=config.min_atr_pct,
        volume_window=config.volume_window,
        min_volume_ratio=config.min_volume_ratio,
        warmup_ltf=config.warmup_ltf,
        warmup_htf=config.warmup_htf,
        max_breakout_atr_extension=config.max_breakout_atr_extension,
        max_entry_rsi=config.max_entry_rsi,
        ema_slope_lookback=config.ema_slope_lookback,
        pullback_ema_buffer_atr=config.pullback_ema_buffer_atr,
    )


def cost_adjusted_price(price: float, fee_bps: float, slippage_bps: float, side: str) -> float:
    adjustment = (fee_bps + slippage_bps) / 10_000.0
    if side == "buy":
        return price * (1.0 + adjustment)
    return price * (1.0 - adjustment)


def compute_metrics(equity_curve: List[float], trades: List[Dict[str, Any]], starting_cash: float) -> Dict[str, Any]:
    if not equity_curve:
        return {
            "starting_cash": starting_cash,
            "ending_equity": starting_cash,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "closed_trades": 0,
            "win_rate": 0.0,
            "avg_trade_pct": 0.0,
        }
    peak = equity_curve[0]
    max_drawdown = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        drawdown = (peak - value) / max(peak, 1e-9)
        max_drawdown = max(max_drawdown, drawdown)
    closed = [trade for trade in trades if trade.get("exit_price") is not None]
    win_rate = sum(1 for trade in closed if trade.get("pnl_value", 0.0) > 0) / max(len(closed), 1)
    avg_trade_pct = sum(float(trade.get("pnl_pct", 0.0)) for trade in closed) / max(len(closed), 1)
    return {
        "starting_cash": starting_cash,
        "ending_equity": equity_curve[-1],
        "total_return_pct": ((equity_curve[-1] / max(starting_cash, 1e-9)) - 1.0) * 100.0,
        "max_drawdown_pct": max_drawdown * 100.0,
        "closed_trades": len(closed),
        "win_rate": win_rate,
        "avg_trade_pct": avg_trade_pct,
    }


def portfolio_runtime_config(config: BacktestConfig) -> Any:
    runtime = as_runtime_config(config)
    runtime.max_entries_per_loop = config.max_entries_per_bar
    runtime.max_portfolio_heat = config.portfolio_cap
    runtime.max_sector_exposure = 1.0
    return runtime


def simulate_strategy(df: pd.DataFrame, config: Optional[BacktestConfig] = None) -> Dict[str, Any]:
    config = config or BacktestConfig()
    runtime = as_runtime_config(config)
    data = normalize_ohlcv(df).copy()
    closes = data["close"].tolist()
    highs = data["high"].tolist()
    lows = data["low"].tolist()
    volumes = data["volume"].tolist()
    index = list(data.index)

    cash = config.cash
    qty = 0.0
    entry_price = 0.0
    entry_source = ""
    entry_regime = ""
    sym_state = {
        "prev_price": None,
        "last_fire_ts": None,
        "tp1_hit": False,
        "warm_ltf_ok": False,
        "warm_htf_ok": False,
        "high_water": None,
        "last_entry_price": None,
        "live_stop": None,
        "entry_source": None,
        "entry_regime": None,
    }
    equity_curve: List[float] = []
    trade_log: List[Dict[str, Any]] = []

    start_idx = max(runtime.warmup_ltf, runtime.warmup_htf)
    for idx in range(start_idx, len(data)):
        window_closes = closes[: idx + 1]
        window_highs = highs[: idx + 1]
        window_lows = lows[: idx + 1]
        window_volumes = volumes[: idx + 1]
        last = window_closes[-1]

        htf_closes = window_closes[-max(runtime.warmup_htf, runtime.htf_slow + runtime.htf_slope_n + 5) :]
        if len(htf_closes) < max(runtime.htf_slow + runtime.htf_slope_n, runtime.warmup_htf):
            equity_curve.append(cash + (qty * last))
            continue

        trend_ok, htf_fast, htf_slow, htf_slope = crypto_trader.compute_trend_ok(runtime, htf_closes)
        trend_value = crypto_trader.trend_score(htf_fast, htf_slow, htf_slope, htf_closes[-1])
        atr_live = crypto_trader.atr(window_highs, window_lows, window_closes, runtime.atr_len)
        atr_pct_live = atr_live / max(last, 1e-9) if atr_live and atr_live == atr_live else 0.0
        vol_ratio_live = crypto_trader.volume_ratio(window_volumes, runtime.volume_window)
        regime = crypto_trader.detect_regime(trend_ok, trend_value, atr_pct_live, vol_ratio_live, None)

        if qty <= 0:
            setup_options = []
            breakout_signal = crypto_trader.build_entry_signal(runtime, window_closes, window_highs, window_lows, window_volumes)
            pullback_signal = crypto_trader.build_pullback_entry_signal(runtime, window_closes, window_highs, window_lows, window_volumes)
            if breakout_signal and regime.allow_breakout:
                setup_options.append(breakout_signal)
            if pullback_signal and regime.allow_pullback:
                setup_options.append(pullback_signal)
            if setup_options:
                signal = sorted(
                    setup_options,
                    key=lambda item: crypto_trader.score_setup(item, trend_value, None, regime),
                    reverse=True,
                )[0]
                qty_size = crypto_trader.size_for_risk(runtime, cash, signal.breakout_level, signal.stop, cash, 0.0)
                qty_size *= signal.confidence * regime.risk_multiplier
                fill_price = cost_adjusted_price(signal.breakout_level, config.fee_bps, config.slippage_bps, "buy")
                notional = qty_size * fill_price
                if qty_size > 0 and notional <= cash:
                    qty = qty_size
                    cash -= notional
                    entry_price = fill_price
                    entry_source = signal.source
                    entry_regime = regime.name
                    sym_state["tp1_hit"] = False
                    sym_state["high_water"] = last
                    sym_state["last_entry_price"] = fill_price
                    sym_state["entry_source"] = entry_source
                    sym_state["entry_regime"] = entry_regime
                    trade_log.append(
                        {
                            "entry_ts": str(index[idx]),
                            "entry_price": entry_price,
                            "exit_price": None,
                            "qty": qty,
                            "source": entry_source,
                            "regime": entry_regime,
                        }
                    )
        else:
            position = crypto_trader.BrokerPosition(
                symbol=config.symbol,
                qty=qty,
                qty_available=qty,
                avg_entry_price=entry_price,
                current_price=last,
                market_value=qty * last,
            )
            stop, tp1, tp2 = crypto_trader.compute_live_exit_levels(runtime, position, sym_state, window_closes, window_highs, window_lows)
            sym_state["live_stop"] = stop
            exit_reason = None
            exit_price = None
            if crypto_trader.crossed_up(sym_state.get("prev_price"), last, tp2):
                exit_reason = "tp2"
                exit_price = cost_adjusted_price(tp2, config.fee_bps, config.slippage_bps, "sell")
            elif crypto_trader.crossed_down(sym_state.get("prev_price"), last, stop):
                exit_reason = "stop"
                exit_price = cost_adjusted_price(stop, config.fee_bps, config.slippage_bps, "sell")
            elif not sym_state.get("tp1_hit", False) and crypto_trader.crossed_up(sym_state.get("prev_price"), last, tp1):
                partial_qty = qty * 0.4
                realized = partial_qty * cost_adjusted_price(tp1, config.fee_bps, config.slippage_bps, "sell")
                cash += realized
                qty -= partial_qty
                sym_state["tp1_hit"] = True
                trade_log.append(
                    {
                        "entry_ts": str(index[idx]),
                        "entry_price": entry_price,
                        "exit_price": tp1,
                        "qty": partial_qty,
                        "source": entry_source,
                        "regime": entry_regime,
                        "pnl_value": (tp1 - entry_price) * partial_qty,
                        "pnl_pct": ((tp1 / max(entry_price, 1e-9)) - 1.0) * 100.0,
                    }
                )

            if exit_reason and exit_price is not None:
                cash += qty * exit_price
                pnl_value = (exit_price - entry_price) * qty
                pnl_pct = ((exit_price / max(entry_price, 1e-9)) - 1.0) * 100.0
                trade_log.append(
                    {
                        "entry_ts": str(index[idx]),
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "qty": qty,
                        "source": entry_source,
                        "regime": entry_regime,
                        "pnl_value": pnl_value,
                        "pnl_pct": pnl_pct,
                        "exit_reason": exit_reason,
                    }
                )
                qty = 0.0
                entry_price = 0.0
                entry_source = ""
                entry_regime = ""
                crypto_trader.reset_trade_state(sym_state)

        sym_state["prev_price"] = last
        equity_curve.append(cash + (qty * last))

    metrics = compute_metrics(equity_curve, trade_log, config.cash)
    return {"metrics": metrics, "equity_curve": equity_curve, "trades": trade_log}


def aligned_history(data_map: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    normalized = {symbol: normalize_ohlcv(df).copy() for symbol, df in data_map.items()}
    common_index = None
    for df in normalized.values():
        common_index = df.index if common_index is None else common_index.intersection(df.index)
    if common_index is None or len(common_index) == 0:
        raise ValueError("No overlapping history across symbols")
    return {symbol: df.loc[common_index].copy() for symbol, df in normalized.items()}


def simulate_portfolio_strategy(data_map: Dict[str, pd.DataFrame], config: Optional[BacktestConfig] = None) -> Dict[str, Any]:
    config = config or BacktestConfig()
    runtime = portfolio_runtime_config(config)
    histories = aligned_history(data_map)
    symbols = list(histories.keys())
    index = list(next(iter(histories.values())).index)
    closes = {symbol: histories[symbol]["close"].tolist() for symbol in symbols}
    highs = {symbol: histories[symbol]["high"].tolist() for symbol in symbols}
    lows = {symbol: histories[symbol]["low"].tolist() for symbol in symbols}
    volumes = {symbol: histories[symbol]["volume"].tolist() for symbol in symbols}

    cash = config.cash
    positions: Dict[str, Dict[str, Any]] = {}
    state = crypto_trader.default_state(config.cash)
    equity_curve: List[float] = []
    trade_log: List[Dict[str, Any]] = []
    start_idx = max(runtime.warmup_ltf, runtime.warmup_htf)

    for idx in range(start_idx, len(index)):
        prices_now = {symbol: closes[symbol][idx] for symbol in symbols}

        for symbol in list(positions.keys()):
            window_closes = closes[symbol][: idx + 1]
            window_highs = highs[symbol][: idx + 1]
            window_lows = lows[symbol][: idx + 1]
            position_data = positions[symbol]
            sym_state = crypto_trader.get_sym_state(state, symbol)
            broker_position = crypto_trader.BrokerPosition(
                symbol=symbol,
                qty=position_data["qty"],
                qty_available=position_data["qty"],
                avg_entry_price=position_data["entry_price"],
                current_price=prices_now[symbol],
                market_value=position_data["qty"] * prices_now[symbol],
            )
            stop, tp1, tp2 = crypto_trader.compute_live_exit_levels(runtime, broker_position, sym_state, window_closes, window_highs, window_lows)
            sym_state["live_stop"] = stop
            prev_price = sym_state.get("prev_price")
            last = prices_now[symbol]
            if crypto_trader.crossed_up(prev_price, last, tp2) or crypto_trader.crossed_down(prev_price, last, stop):
                reason = "tp2" if crypto_trader.crossed_up(prev_price, last, tp2) else "stop"
                exit_level = tp2 if reason == "tp2" else stop
                exit_price = cost_adjusted_price(exit_level, config.fee_bps, config.slippage_bps, "sell")
                qty = position_data["qty"]
                cash += qty * exit_price
                trade_log.append(
                    {
                        "entry_ts": position_data["entry_ts"],
                        "exit_ts": str(index[idx]),
                        "symbol": symbol,
                        "entry_price": position_data["entry_price"],
                        "exit_price": exit_price,
                        "qty": qty,
                        "source": position_data["source"],
                        "regime": position_data["regime"],
                        "pnl_value": (exit_price - position_data["entry_price"]) * qty,
                        "pnl_pct": ((exit_price / max(position_data["entry_price"], 1e-9)) - 1.0) * 100.0,
                        "exit_reason": reason,
                    }
                )
                del positions[symbol]
                crypto_trader.reset_trade_state(sym_state)
            elif not sym_state.get("tp1_hit", False) and crypto_trader.crossed_up(prev_price, last, tp1):
                partial_qty = position_data["qty"] * 0.4
                fill_price = cost_adjusted_price(tp1, config.fee_bps, config.slippage_bps, "sell")
                cash += partial_qty * fill_price
                position_data["qty"] -= partial_qty
                sym_state["tp1_hit"] = True
                trade_log.append(
                    {
                        "entry_ts": position_data["entry_ts"],
                        "exit_ts": str(index[idx]),
                        "symbol": symbol,
                        "entry_price": position_data["entry_price"],
                        "exit_price": fill_price,
                        "qty": partial_qty,
                        "source": position_data["source"],
                        "regime": position_data["regime"],
                        "pnl_value": (fill_price - position_data["entry_price"]) * partial_qty,
                        "pnl_pct": ((fill_price / max(position_data["entry_price"], 1e-9)) - 1.0) * 100.0,
                        "exit_reason": "tp1",
                    }
                )
            sym_state["prev_price"] = last

        candidates: List[crypto_trader.CandidateSetup] = []
        allocated_now = sum(pos["qty"] * prices_now[symbol] for symbol, pos in positions.items())
        account_equity = cash + sum(pos["qty"] * prices_now[symbol] for symbol, pos in positions.items())
        for symbol in symbols:
            if symbol in positions:
                continue
            window_closes = closes[symbol][: idx + 1]
            window_highs = highs[symbol][: idx + 1]
            window_lows = lows[symbol][: idx + 1]
            window_volumes = volumes[symbol][: idx + 1]
            htf_closes = window_closes[-max(runtime.warmup_htf, runtime.htf_slow + runtime.htf_slope_n + 5) :]
            if len(htf_closes) < max(runtime.htf_slow + runtime.htf_slope_n, runtime.warmup_htf):
                continue
            trend_ok, htf_fast, htf_slow, htf_slope = crypto_trader.compute_trend_ok(runtime, htf_closes)
            trend_value = crypto_trader.trend_score(htf_fast, htf_slow, htf_slope, htf_closes[-1])
            atr_live = crypto_trader.atr(window_highs, window_lows, window_closes, runtime.atr_len)
            atr_pct_live = atr_live / max(window_closes[-1], 1e-9) if atr_live and atr_live == atr_live else 0.0
            vol_ratio_live = crypto_trader.volume_ratio(window_volumes, runtime.volume_window)
            regime = crypto_trader.detect_regime(trend_ok, trend_value, atr_pct_live, vol_ratio_live, None)
            setup_options = []
            breakout_signal = crypto_trader.build_entry_signal(runtime, window_closes, window_highs, window_lows, window_volumes)
            pullback_signal = crypto_trader.build_pullback_entry_signal(runtime, window_closes, window_highs, window_lows, window_volumes)
            if breakout_signal and regime.allow_breakout:
                setup_options.append(breakout_signal)
            if pullback_signal and regime.allow_pullback:
                setup_options.append(pullback_signal)
            if not setup_options:
                continue
            signal = sorted(
                setup_options,
                key=lambda item: crypto_trader.score_setup(item, trend_value, None, regime),
                reverse=True,
            )[0]
            candidates.append(
                crypto_trader.CandidateSetup(
                    symbol=symbol,
                    signal=signal,
                    regime=regime,
                    sentiment=None,
                    trend_score=trend_value,
                    sector=crypto_trader.sector_for_symbol(symbol),
                    score=crypto_trader.score_setup(signal, trend_value, None, regime),
                    last_price=window_closes[-1],
                )
            )

        for candidate in sorted(candidates, key=lambda item: item.score, reverse=True)[: config.max_entries_per_bar]:
            qty_size = crypto_trader.size_for_risk(
                runtime,
                cash,
                candidate.signal.breakout_level,
                candidate.signal.stop,
                account_equity,
                allocated_now,
            )
            qty_size *= candidate.signal.confidence * candidate.regime.risk_multiplier
            fill_price = cost_adjusted_price(candidate.signal.breakout_level, config.fee_bps, config.slippage_bps, "buy")
            notional = qty_size * fill_price
            if qty_size <= 0 or notional > cash:
                continue
            positions[candidate.symbol] = {
                "qty": qty_size,
                "entry_price": fill_price,
                "entry_ts": str(index[idx]),
                "source": candidate.signal.source,
                "regime": candidate.regime.name,
            }
            sym_state = crypto_trader.get_sym_state(state, candidate.symbol)
            sym_state["tp1_hit"] = False
            sym_state["high_water"] = candidate.last_price
            sym_state["last_entry_price"] = fill_price
            sym_state["entry_source"] = candidate.signal.source
            sym_state["entry_regime"] = candidate.regime.name
            cash -= notional
            allocated_now += notional

        equity_curve.append(cash + sum(pos["qty"] * prices_now[symbol] for symbol, pos in positions.items()))

    metrics = compute_metrics(equity_curve, trade_log, config.cash)
    metrics["symbols"] = symbols
    metrics["open_positions"] = len(positions)
    return {"metrics": metrics, "equity_curve": equity_curve, "trades": trade_log}


def walk_forward_validate(df: pd.DataFrame, config: Optional[BacktestConfig] = None) -> Dict[str, Any]:
    config = config or BacktestConfig()
    data = normalize_ohlcv(df)
    if len(data) < config.train_bars + config.test_bars:
        result = simulate_strategy(data, config)
        return {"windows": [result["metrics"]], "summary": result["metrics"]}

    windows: List[Dict[str, Any]] = []
    start = 0
    while start + config.train_bars + config.test_bars <= len(data):
        window = data.iloc[start : start + config.train_bars + config.test_bars]
        test_slice = window.iloc[config.train_bars :]
        result = simulate_strategy(test_slice, config)
        metrics = dict(result["metrics"])
        metrics["start"] = str(test_slice.index[0])
        metrics["end"] = str(test_slice.index[-1])
        windows.append(metrics)
        start += config.test_bars

    summary = {
        "windows": len(windows),
        "avg_return_pct": sum(item["total_return_pct"] for item in windows) / max(len(windows), 1),
        "avg_max_drawdown_pct": sum(item["max_drawdown_pct"] for item in windows) / max(len(windows), 1),
        "avg_win_rate": sum(item["win_rate"] for item in windows) / max(len(windows), 1),
    }
    return {"windows": windows, "summary": summary}


def run(symbol: str = "SPY", start: str = "2018-01-01", cash: float = 100_000) -> float:
    config = BacktestConfig(symbol=symbol, start=start, cash=cash)
    df = load_price_history(symbol, start, config.interval)
    result = simulate_strategy(df, config)
    final_equity = round(float(result["metrics"]["ending_equity"]), 2)
    print(
        f"Backtest {symbol}: ending=${final_equity:.2f} "
        f"return={result['metrics']['total_return_pct']:.2f}% "
        f"max_dd={result['metrics']['max_drawdown_pct']:.2f}% "
        f"trades={result['metrics']['closed_trades']}"
    )
    return final_equity


if __name__ == "__main__":
    run()
