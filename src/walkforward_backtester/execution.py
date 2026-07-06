"""Event-driven execution simulator: turns target-position signals into
fills, an equity curve and a trade log with realistic frictions.

Timing convention: a signal computed from data available through the close
of day ``t`` is not actionable until the next bar. The position decided at
``t`` is therefore entered/adjusted at the **open of day t+1**, and marked to
market using close-to-close returns from then on. This is enforced here
(strategies never see their own shifted output) so no strategy can
accidentally look ahead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ExecutionConfig:
    initial_capital: float = 100_000.0
    commission_bps: float = 5.0
    slippage_bps: float = 5.0
    slippage_model: str = "bps"  # "bps" or "vol_scaled"
    vol_scale_lookback: int = 20
    position_sizing: str = "fixed_fraction"  # "fixed_fraction" or "vol_target"
    fixed_fraction: float = 0.95
    vol_target_annual: float = 0.15
    vol_lookback: int = 20
    max_gross_exposure: float = 1.0
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None

    def __post_init__(self) -> None:
        if self.position_sizing not in {"fixed_fraction", "vol_target"}:
            raise ValueError("position_sizing must be 'fixed_fraction' or 'vol_target'")
        if self.slippage_model not in {"bps", "vol_scaled"}:
            raise ValueError("slippage_model must be 'bps' or 'vol_scaled'")


@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    side: int
    shares: float
    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    commission_paid: float = 0.0
    slippage_paid: float = 0.0

    @property
    def pnl(self) -> float | None:
        if self.exit_price is None:
            return None
        gross = (self.exit_price - self.entry_price) * self.side * self.shares
        return gross - self.commission_paid - self.slippage_paid

    @property
    def pnl_pct(self) -> float | None:
        if self.exit_price is None:
            return None
        return (self.exit_price / self.entry_price - 1.0) * self.side

    def to_dict(self) -> dict:
        return {
            "entry_date": self.entry_date,
            "entry_price": self.entry_price,
            "side": self.side,
            "shares": self.shares,
            "exit_date": self.exit_date,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "commission_paid": self.commission_paid,
            "slippage_paid": self.slippage_paid,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
        }


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    daily_returns: pd.Series
    trades: list[Trade]
    positions: pd.Series

    def trade_log(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(
                columns=[
                    "entry_date", "entry_price", "side", "shares", "exit_date",
                    "exit_price", "exit_reason", "commission_paid", "slippage_paid",
                    "pnl", "pnl_pct",
                ]
            )
        return pd.DataFrame([t.to_dict() for t in self.trades])


def _slippage_bps(config: ExecutionConfig, df: pd.DataFrame, i: int) -> float:
    if config.slippage_model == "bps":
        return config.slippage_bps
    returns = df["Close"].pct_change()
    lookback = returns.iloc[max(0, i - config.vol_scale_lookback) : i]
    if lookback.empty or lookback.std(ddof=0) == 0 or np.isnan(lookback.std(ddof=0)):
        return config.slippage_bps
    daily_vol = lookback.std(ddof=0)
    # Baseline reference vol: an early-period window, but capped at `i` so
    # that for bars near the start of the series it never reaches past the
    # current bar into data that wouldn't yet be known (i.e. it stays a
    # strict prefix of history-up-to-i, not a fixed early window that could
    # overlap the future when i is small).
    baseline_vol = returns.iloc[: min(max(config.vol_scale_lookback, 1), i)].std(ddof=0)
    if not baseline_vol or np.isnan(baseline_vol) or baseline_vol == 0:
        return config.slippage_bps
    return config.slippage_bps * float(daily_vol / baseline_vol)


def _size_position(
    config: ExecutionConfig, capital: float, price: float, df: pd.DataFrame, i: int
) -> float:
    if config.position_sizing == "fixed_fraction":
        notional = capital * config.fixed_fraction
    else:
        returns = df["Close"].pct_change()
        lookback = returns.iloc[max(0, i - config.vol_lookback) : i]
        daily_vol = lookback.std(ddof=0)
        if not daily_vol or np.isnan(daily_vol) or daily_vol == 0:
            notional = 0.0
        else:
            ann_vol = daily_vol * np.sqrt(252)
            leverage = config.vol_target_annual / ann_vol
            notional = capital * leverage
    notional = min(notional, capital * config.max_gross_exposure)
    notional = max(notional, 0.0)
    return notional / price if price > 0 else 0.0


def run_backtest(
    df: pd.DataFrame, target_position: pd.Series, config: ExecutionConfig | None = None
) -> BacktestResult:
    """Simulate execution of ``target_position`` (as produced by a
    ``Strategy``) against OHLC bars in ``df``, with next-bar fills,
    commission, slippage, position sizing, and optional stop-loss /
    take-profit.
    """
    config = config or ExecutionConfig()
    df = df.loc[target_position.index]
    n = len(df)
    dates = df.index

    desired = target_position.shift(1).fillna(0)  # actionable position for day i is decided at close of i-1

    capital = config.initial_capital
    equity = np.empty(n)
    cash = capital
    shares_held = 0.0
    side_held = 0
    current_trade: Trade | None = None
    trades: list[Trade] = []
    realized_positions = np.zeros(n, dtype=int)

    for i in range(n):
        date = dates[i]
        open_px = df["Open"].iloc[i]
        high_px = df["High"].iloc[i]
        low_px = df["Low"].iloc[i]
        close_px = df["Close"].iloc[i]
        want = int(desired.iloc[i])

        # 1. Check stop-loss / take-profit intrabar before honoring new signals.
        forced_exit_price = None
        forced_reason = None
        if current_trade is not None:
            entry_price = current_trade.entry_price
            if side_held == 1:
                if config.stop_loss_pct and low_px <= entry_price * (1 - config.stop_loss_pct):
                    forced_exit_price = entry_price * (1 - config.stop_loss_pct)
                    forced_reason = "stop_loss"
                elif config.take_profit_pct and high_px >= entry_price * (1 + config.take_profit_pct):
                    forced_exit_price = entry_price * (1 + config.take_profit_pct)
                    forced_reason = "take_profit"
            elif side_held == -1:
                if config.stop_loss_pct and high_px >= entry_price * (1 + config.stop_loss_pct):
                    forced_exit_price = entry_price * (1 + config.stop_loss_pct)
                    forced_reason = "stop_loss"
                elif config.take_profit_pct and low_px <= entry_price * (1 - config.take_profit_pct):
                    forced_exit_price = entry_price * (1 - config.take_profit_pct)
                    forced_reason = "take_profit"

        if forced_exit_price is not None and current_trade is not None:
            slip_bps = _slippage_bps(config, df, i)
            fill = forced_exit_price * (1 - side_held * slip_bps / 10_000)
            commission = abs(fill * shares_held) * config.commission_bps / 10_000
            slippage_cost = abs(forced_exit_price - fill) * shares_held
            current_trade.exit_date = date
            current_trade.exit_price = fill
            current_trade.exit_reason = forced_reason
            current_trade.commission_paid += commission
            current_trade.slippage_paid += slippage_cost
            cash += fill * shares_held * side_held - commission
            trades.append(current_trade)
            current_trade = None
            shares_held = 0.0
            side_held = 0
            want = 0  # flat for the remainder of this bar once stopped out

        # 2. Honor signal-driven position changes at today's open.
        if want != side_held:
            if current_trade is not None:
                slip_bps = _slippage_bps(config, df, i)
                fill = open_px * (1 - side_held * slip_bps / 10_000)
                commission = abs(fill * shares_held) * config.commission_bps / 10_000
                slippage_cost = abs(open_px - fill) * shares_held
                current_trade.exit_date = date
                current_trade.exit_price = fill
                current_trade.exit_reason = "signal"
                current_trade.commission_paid += commission
                current_trade.slippage_paid += slippage_cost
                cash += fill * shares_held * side_held - commission
                trades.append(current_trade)
                current_trade = None
                shares_held = 0.0
                side_held = 0

            if want != 0:
                mark_to_market_capital = cash
                shares = _size_position(config, mark_to_market_capital, open_px, df, i)
                slip_bps = _slippage_bps(config, df, i)
                fill = open_px * (1 + want * slip_bps / 10_000)
                commission = abs(fill * shares) * config.commission_bps / 10_000
                slippage_cost = abs(open_px - fill) * shares
                cash -= fill * shares * want + commission
                current_trade = Trade(
                    entry_date=date,
                    entry_price=fill,
                    side=want,
                    shares=shares,
                    commission_paid=commission,
                    slippage_paid=slippage_cost,
                )
                shares_held = shares
                side_held = want

        realized_positions[i] = side_held
        equity[i] = cash + shares_held * side_held * close_px

    if current_trade is not None:
        current_trade.exit_date = dates[-1]
        current_trade.exit_price = df["Close"].iloc[-1]
        current_trade.exit_reason = "end_of_data"
        trades.append(current_trade)

    equity_curve = pd.Series(equity, index=dates, name="equity")
    daily_returns = equity_curve.pct_change().fillna(0.0)
    positions = pd.Series(realized_positions, index=dates, name="position")
    return BacktestResult(equity_curve=equity_curve, daily_returns=daily_returns, trades=trades, positions=positions)
