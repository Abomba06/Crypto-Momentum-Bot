def notional_from_cash(cash: float, risk_pct: float) -> float:
    return max(10.0, round(cash * risk_pct, 2))

def clamp_position(target_notional: float, max_portfolio_pct: float, equity: float) -> float:
    cap = equity * max_portfolio_pct
    return min(target_notional, cap)
