"""Post-hoc regime analysis: how does each strategy's realized Sharpe differ
in high/low volatility and trending/sideways markets?

This is purely descriptive and computed after the walk-forward test returns
already exist; it does not feed back into parameter selection, so it cannot
introduce look-ahead bias into the walk-forward results themselves.
"""

from __future__ import annotations

import pandas as pd

from . import indicators as ind
from .metrics import sharpe_ratio


def volatility_regime(price: pd.DataFrame, window: int = 20) -> pd.Series:
    returns = price["Close"].pct_change()
    vol = ind.realized_vol(returns, window=window)
    valid = vol.dropna()
    if valid.empty:
        return pd.Series(index=price.index, dtype=object)
    low_cut, high_cut = valid.quantile([1 / 3, 2 / 3])

    def label(v: float) -> str:
        if pd.isna(v):
            return "unknown"
        if v <= low_cut:
            return "low_vol"
        if v >= high_cut:
            return "high_vol"
        return "mid_vol"

    return vol.map(label).rename("vol_regime")


def trend_regime(price: pd.DataFrame, window: int = 60, threshold: float = 0.35) -> pd.Series:
    er = ind.efficiency_ratio(price["Close"], window=window)

    def label(v: float) -> str:
        if pd.isna(v):
            return "unknown"
        return "trending" if v >= threshold else "sideways"

    return er.map(label).rename("trend_regime")


def sharpe_by_regime(daily_returns: pd.Series, regime: pd.Series) -> pd.Series:
    aligned = pd.concat([daily_returns.rename("ret"), regime.rename("regime")], axis=1).dropna()
    out = {}
    for label, group in aligned.groupby("regime"):
        if label == "unknown":
            continue
        out[label] = sharpe_ratio(group["ret"])
    return pd.Series(out)
