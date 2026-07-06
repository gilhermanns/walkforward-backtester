"""Pluggable trading strategies sharing a common ``generate_signals`` interface.

Every strategy consumes a daily OHLCV DataFrame and returns a target-position
series indexed the same way, using only information available up to and
including each row's own close. Turning that into trades with a next-bar fill
is the execution engine's job (see ``execution.py``), not the strategy's.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import product
from typing import Any

import pandas as pd

from . import indicators as ind


class Strategy(ABC):
    """Common interface for all strategies."""

    name: str = "strategy"

    def __init__(self, **params: Any) -> None:
        self.params = params

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return a target position series in {-1, 0, 1} (or a subset),
        indexed like ``df``, using only data available through each row.
        """

    @classmethod
    @abstractmethod
    def param_grid(cls) -> list[dict[str, Any]]:
        """Enumerate the parameter combinations searched during walk-forward
        validation."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        params = ", ".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.name}({params})"


class MovingAverageCrossover(Strategy):
    """Long when the fast SMA is above the slow SMA, flat (or short)
    otherwise."""

    name = "ma_crossover"

    def __init__(self, fast: int = 20, slow: int = 100, allow_short: bool = False) -> None:
        if fast >= slow:
            raise ValueError("fast window must be shorter than slow window")
        super().__init__(fast=fast, slow=slow, allow_short=allow_short)
        self.fast = fast
        self.slow = slow
        self.allow_short = allow_short

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        fast_ma = ind.sma(df["Close"], self.fast)
        slow_ma = ind.sma(df["Close"], self.slow)
        long_flag = fast_ma > slow_ma
        if self.allow_short:
            position = long_flag.astype(int) - (~long_flag).astype(int)
        else:
            position = long_flag.astype(int)
        position[fast_ma.isna() | slow_ma.isna()] = 0
        return position.rename("position")

    @classmethod
    def param_grid(cls) -> list[dict[str, Any]]:
        combos = []
        for fast, slow in product([5, 10, 20, 30], [50, 100,150, 200]):
            if fast < slow:
                combos.append({"fast": fast, "slow": slow})
        return combos


class RSIMeanReversion(Strategy):
    """Buy oversold dips, exit once RSI recovers past the neutral line;
    optionally mirror the logic on the short side for overbought spikes."""

    name = "rsi_mean_reversion"

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        exit_neutral: float = 50.0,
        allow_short: bool = False,
    ) -> None:
        super().__init__(
            period=period,
            oversold=oversold,
            overbought=overbought,
            exit_neutral=exit_neutral,
            allow_short=allow_short,
        )
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.exit_neutral = exit_neutral
        self.allow_short = allow_short

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        rsi = ind.rsi(df["Close"], self.period)
        position = pd.Series(0, index=df.index, dtype=int)
        state = 0
        for i, value in enumerate(rsi.to_numpy()):
            if pd.isna(value):
                position.iloc[i] = 0
                continue
            if state == 0 and value < self.oversold:
                state = 1
            elif state == 0 and self.allow_short and value > self.overbought:
                state = -1
            elif state == 1 and value >= self.exit_neutral:
                state = 0
            elif state == -1 and value <= self.exit_neutral:
                state = 0
            position.iloc[i] = state
        return position.rename("position")

    @classmethod
    def param_grid(cls) -> list[dict[str, Any]]:
        combos = []
        for period, oversold, overbought in product([7, 14, 21], [20, 25, 30], [70, 75, 80]):
            combos.append(
                {"period": period, "oversold": oversold, "overbought": overbought, "exit_neutral": 50.0}
            )
        return combos


class DonchianBreakout(Strategy):
    """Classic N-day high/low channel breakout (Turtle-style)."""

    name = "donchian_breakout"

    def __init__(self, entry_window: int = 20, exit_window: int = 10, allow_short: bool = False) -> None:
        if exit_window > entry_window:
            raise ValueError("exit_window should not exceed entry_window")
        super().__init__(entry_window=entry_window, exit_window=exit_window, allow_short=allow_short)
        self.entry_window = entry_window
        self.exit_window = exit_window
        self.allow_short = allow_short

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        upper = ind.donchian_high(df["High"], self.entry_window).shift(1)
        lower = ind.donchian_low(df["Low"], self.entry_window).shift(1)
        exit_upper = ind.donchian_high(df["High"], self.exit_window).shift(1)
        exit_lower = ind.donchian_low(df["Low"], self.exit_window).shift(1)
        close = df["Close"]

        position = pd.Series(0, index=df.index, dtype=int)
        state = 0
        for i in range(len(df)):
            if pd.isna(upper.iloc[i]) or pd.isna(lower.iloc[i]):
                position.iloc[i] = 0
                continue
            price = close.iloc[i]
            if state == 0 and price > upper.iloc[i]:
                state = 1
            elif state == 0 and self.allow_short and price < lower.iloc[i]:
                state = -1
            elif state == 1 and price < exit_lower.iloc[i]:
                state = 0
            elif state == -1 and price > exit_upper.iloc[i]:
                state = 0
            position.iloc[i] = state
        return position.rename("position")

    @classmethod
    def param_grid(cls) -> list[dict[str, Any]]:
        combos = []
        for entry_window, exit_window in product([10, 20, 55], [5, 10, 20]):
            if exit_window <= entry_window:
                combos.append({"entry_window": entry_window, "exit_window": exit_window})
        return combos


STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    MovingAverageCrossover.name: MovingAverageCrossover,
    RSIMeanReversion.name: RSIMeanReversion,
    DonchianBreakout.name: DonchianBreakout,
}


class BuyAndHold(Strategy):
    """Benchmark: fully invested from the first valid bar onward."""

    name = "buy_and_hold"

    def __init__(self) -> None:
        super().__init__()

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(1, index=df.index, dtype=int).rename("position")

    @classmethod
    def param_grid(cls) -> list[dict[str, Any]]:
        return [{}]
