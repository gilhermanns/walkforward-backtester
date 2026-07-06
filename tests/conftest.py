import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def crossover_ohlc() -> pd.DataFrame:
    """Synthetic series engineered so a 3-day SMA crosses a 6-day SMA up
    around day 10 and back down around day 22 -- deterministic, known
    crossover points for testing MovingAverageCrossover."""
    n = 40
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = np.concatenate(
        [
            np.linspace(100, 100, 8),
            np.linspace(100, 130, 12),
            np.linspace(130, 90, 12),
            np.linspace(90, 90, n - 32),
        ]
    )
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": prices + 0.5,
            "Low": prices - 0.5,
            "Close": prices,
            "Volume": 1_000_000,
        },
        index=dates,
    )
    return df


@pytest.fixture
def rsi_ohlc() -> pd.DataFrame:
    """Synthetic series with a sharp drop (drives RSI below 30) followed by
    a recovery (drives RSI back above 50), for RSIMeanReversion tests."""
    n = 60
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = [100.0]
    for i in range(1, n):
        if 20 <= i < 30:
            prices.append(prices[-1] * 0.97)
        elif 30 <= i < 45:
            prices.append(prices[-1] * 1.03)
        else:
            prices.append(prices[-1] * (1 + 0.0005 * ((-1) ** i)))
    prices = np.array(prices)
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": prices * 1.002,
            "Low": prices * 0.998,
            "Close": prices,
            "Volume": 1_000_000,
        },
        index=dates,
    )
    return df


@pytest.fixture
def breakout_ohlc() -> pd.DataFrame:
    """Flat range for 15 bars, then a clean breakout above the prior
    20-day high, for DonchianBreakout tests."""
    n = 45
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = np.concatenate(
        [
            100 + np.sin(np.linspace(0, 6, 20)) * 1.0,
            np.linspace(101, 140, 15),
            np.linspace(140, 140, n - 35),
        ]
    )
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": prices + 1.0,
            "Low": prices - 1.0,
            "Close": prices,
            "Volume": 1_000_000,
        },
        index=dates,
    )
    return df


@pytest.fixture
def toy_trade_df() -> pd.DataFrame:
    """A 5-day OHLC frame with a single clean long round-trip, used to check
    commission/slippage arithmetic by hand."""
    dates = pd.date_range("2021-01-04", periods=5, freq="B")
    data = {
        "Open": [100.0, 100.0, 110.0, 110.0, 110.0],
        "High": [100.0, 105.0, 111.0, 110.0, 110.0],
        "Low": [100.0, 99.0, 109.0, 110.0, 110.0],
        "Close": [100.0, 104.0, 110.0, 110.0, 110.0],
        "Volume": [1_000_000] * 5,
    }
    return pd.DataFrame(data, index=dates)
