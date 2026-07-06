"""Rolling train / validation / test walk-forward framework.

Each fold has three contiguous, non-overlapping segments:

    [train_start, train_end) [val_start=train_end, val_end) [test_start=val_end, test_end)

``train`` only ever serves as an indicator warm-up buffer (these strategies
have no statistical fitting step). Parameters are grid-searched and selected
by maximizing Sharpe ratio on ``validation`` alone; the winning parameters are
then evaluated, completely untouched, on ``test``. The splitter is pure date
arithmetic and never looks at the price data, so there is no way for a later
fold's data to leak into an earlier decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .execution import BacktestResult, ExecutionConfig, run_backtest
from .metrics import sharpe_ratio, summarize
from .strategies import Strategy


@dataclass(frozen=True)
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class WalkForwardSplitter:
    def __init__(
        self,
        train_period: pd.DateOffset = pd.DateOffset(years=2),
        val_period: pd.DateOffset = pd.DateOffset(years=1),
        test_period: pd.DateOffset = pd.DateOffset(months=6),
        step: pd.DateOffset | None = None,
    ) -> None:
        self.train_period = train_period
        self.val_period = val_period
        self.test_period = test_period
        self.step = step or test_period

    def split(self, index: pd.DatetimeIndex) -> list[Fold]:
        if len(index) == 0:
            return []
        start = index[0]
        end = index[-1]
        folds: list[Fold] = []
        train_start = start
        while True:
            train_end = train_start + self.train_period
            val_end = train_end + self.val_period
            test_end = val_end + self.test_period
            if test_end > end:
                break
            folds.append(
                Fold(
                    train_start=train_start,
                    train_end=train_end,
                    val_start=train_end,
                    val_end=val_end,
                    test_start=val_end,
                    test_end=test_end,
                )
            )
            train_start = train_start + self.step
        return folds


def _slice(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df.index >= start) & (df.index < end)]


def _segment_result(full_signals: pd.Series, df: pd.DataFrame, seg_start, seg_end, config: ExecutionConfig) -> BacktestResult:
    """Backtest restricted to [seg_start, seg_end) using signals computed on
    the wider history (so indicators are warmed up), resetting capital at the
    segment boundary so folds are independent, comparable capital
    allocations."""
    seg_df = _slice(df, seg_start, seg_end)
    seg_signals = full_signals.loc[seg_df.index]
    return run_backtest(seg_df, seg_signals, config)


def select_best_params(
    strategy_cls: type[Strategy],
    df: pd.DataFrame,
    fold: Fold,
    config: ExecutionConfig,
) -> tuple[dict, float]:
    history = _slice(df, fold.train_start, fold.val_end)
    best_params: dict = {}
    best_sharpe = float("-inf")
    for params in strategy_cls.param_grid():
        strategy = strategy_cls(**params)
        signals = strategy.generate_signals(history)
        result = _segment_result(signals, df, fold.val_start, fold.val_end, config)
        score = sharpe_ratio(result.daily_returns)
        if score > best_sharpe:
            best_sharpe = score
            best_params = params
    return best_params, best_sharpe


@dataclass
class FoldOutcome:
    fold: Fold
    best_params: dict
    validation_sharpe: float
    test_result: BacktestResult


def run_walkforward(
    strategy_cls: type[Strategy],
    df: pd.DataFrame,
    splitter: WalkForwardSplitter,
    config: ExecutionConfig | None = None,
) -> list[FoldOutcome]:
    config = config or ExecutionConfig()
    folds = splitter.split(df.index)
    outcomes: list[FoldOutcome] = []
    for fold in folds:
        best_params, val_sharpe = select_best_params(strategy_cls, df, fold, config)
        history = _slice(df, fold.train_start, fold.test_end)
        strategy = strategy_cls(**best_params)
        signals = strategy.generate_signals(history)
        test_result = _segment_result(signals, df, fold.test_start, fold.test_end, config)
        outcomes.append(
            FoldOutcome(fold=fold, best_params=best_params, validation_sharpe=val_sharpe, test_result=test_result)
        )
    return outcomes


def stitch_oos_curve(outcomes: list[FoldOutcome], initial_capital: float) -> pd.Series:
    """Chain each fold's out-of-sample daily returns into one continuous
    compounded equity curve, as if capital were rolled from one test window
    straight into the next."""
    pieces = []
    for outcome in outcomes:
        rets = outcome.test_result.daily_returns
        if len(rets) == 0:
            continue
        pieces.append(rets)
    if not pieces:
        return pd.Series(dtype=float)
    all_returns = pd.concat(pieces).sort_index()
    all_returns = all_returns[~all_returns.index.duplicated(keep="first")]
    equity = (1 + all_returns).cumprod() * initial_capital
    return equity.rename("equity")


def stitch_trade_log(outcomes: list[FoldOutcome]) -> pd.DataFrame:
    frames = [o.test_result.trade_log() for o in outcomes if len(o.test_result.trades) > 0]
    if not frames:
        return outcomes[0].test_result.trade_log() if outcomes else pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("entry_date").reset_index(drop=True)


def walkforward_summary(outcomes: list[FoldOutcome], initial_capital: float) -> dict:
    equity = stitch_oos_curve(outcomes, initial_capital)
    trades = stitch_trade_log(outcomes)
    daily_returns = equity.pct_change().fillna(0.0)
    return summarize(equity, daily_returns, trades)
