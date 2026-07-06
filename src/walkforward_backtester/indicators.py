"""Small set of dependency-free technical indicators used by the strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))
    out[avg_loss == 0] = 100.0
    return out


def donchian_high(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).max()


def donchian_low(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).min()


def realized_vol(returns: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
    vol = returns.rolling(window, min_periods=window).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def efficiency_ratio(price: pd.Series, window: int = 60) -> pd.Series:
    """Kaufman efficiency ratio: net move over the window divided by the sum
    of absolute daily moves. Close to 1 => strongly trending, close to 0 =>
    choppy/sideways.
    """
    net_change = (price - price.shift(window)).abs()
    path_length = price.diff().abs().rolling(window, min_periods=window).sum()
    return (net_change / path_length.replace(0.0, np.nan)).clip(0, 1)
