"""Performance metrics computed from an equity curve / trade log."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cagr(equity_curve: pd.Series) -> float:
    if len(equity_curve) < 2 or equity_curve.iloc[0] <= 0:
        return 0.0
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    if total_return <= 0:
        return -1.0
    return total_return ** (1 / years) - 1


def annualized_vol(daily_returns: pd.Series) -> float:
    if daily_returns.std(ddof=0) == 0 or daily_returns.empty:
        return 0.0
    return float(daily_returns.std(ddof=0) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(daily_returns: pd.Series, risk_free: float = 0.0) -> float:
    excess = daily_returns - risk_free / TRADING_DAYS
    std = excess.std(ddof=0)
    if std == 0 or np.isnan(std) or excess.empty:
        return 0.0
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS))


def max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    running_max = equity_curve.cummax()
    return equity_curve / running_max - 1.0


def calmar_ratio(equity_curve: pd.Series) -> float:
    mdd = max_drawdown(equity_curve)
    if mdd == 0:
        return 0.0
    return cagr(equity_curve) / abs(mdd)


def win_rate(trade_pnls: pd.Series) -> float:
    trade_pnls = trade_pnls.dropna()
    if trade_pnls.empty:
        return 0.0
    return float((trade_pnls > 0).mean())


def profit_factor(trade_pnls: pd.Series) -> float:
    trade_pnls = trade_pnls.dropna()
    gains = trade_pnls[trade_pnls > 0].sum()
    losses = -trade_pnls[trade_pnls < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def summarize(equity_curve: pd.Series, daily_returns: pd.Series, trade_log: pd.DataFrame) -> dict:
    pnls = trade_log["pnl"] if "pnl" in trade_log.columns else pd.Series(dtype=float)
    return {
        "total_return": float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) if len(equity_curve) > 1 else 0.0,
        "cagr": cagr(equity_curve),
        "annualized_vol": annualized_vol(daily_returns),
        "sharpe": sharpe_ratio(daily_returns),
        "max_drawdown": max_drawdown(equity_curve),
        "calmar": calmar_ratio(equity_curve),
        "num_trades": int(len(pnls.dropna())),
        "win_rate": win_rate(pnls),
        "profit_factor": profit_factor(pnls),
        "avg_trade_pnl": float(pnls.dropna().mean()) if not pnls.dropna().empty else 0.0,
    }
