import numpy as np
import pandas as pd
import pytest

from walkforward_backtester.strategies import (
    BuyAndHold,
    DonchianBreakout,
    MovingAverageCrossover,
    RSIMeanReversion,
)


def test_ma_crossover_goes_long_when_fast_crosses_above_slow(crossover_ohlc):
    strat = MovingAverageCrossover(fast=3, slow=6)
    pos = strat.generate_signals(crossover_ohlc)

    assert pos.loc["2020-01-13"] == 0
    assert pos.loc["2020-01-14"] == 1
    assert pos.loc["2020-01-30"] == 1
    assert pos.loc["2020-01-31"] == 0


def test_ma_crossover_flat_during_indicator_warmup(crossover_ohlc):
    strat = MovingAverageCrossover(fast=3, slow=6)
    pos = strat.generate_signals(crossover_ohlc)
    assert (pos.iloc[:5] == 0).all()


def test_ma_crossover_rejects_fast_ge_slow():
    with pytest.raises(ValueError):
        MovingAverageCrossover(fast=50, slow=20)


def test_ma_crossover_allow_short_goes_negative(crossover_ohlc):
    strat = MovingAverageCrossover(fast=3, slow=6, allow_short=True)
    pos = strat.generate_signals(crossover_ohlc)
    assert pos.loc["2020-01-31"] == -1


def test_rsi_enters_on_oversold_and_exits_on_neutral_cross(rsi_ohlc):
    strat = RSIMeanReversion(period=14, oversold=30, overbought=70, exit_neutral=50)
    pos = strat.generate_signals(rsi_ohlc)

    assert pos.loc["2020-01-28"] == 0
    assert pos.loc["2020-01-29"] == 1
    assert pos.loc["2020-02-19"] == 1
    assert pos.loc["2020-02-20"] == 0


def test_rsi_never_enters_before_period_warmup(rsi_ohlc):
    strat = RSIMeanReversion(period=14)
    pos = strat.generate_signals(rsi_ohlc)
    assert (pos.iloc[:14] == 0).all()


def test_rsi_allow_short_enters_on_overbought(rsi_ohlc):
    strat = RSIMeanReversion(period=14, oversold=30, overbought=70, exit_neutral=50, allow_short=True)
    pos = strat.generate_signals(rsi_ohlc)
    assert (pos == -1).any()


def test_donchian_breakout_triggers_on_new_high(breakout_ohlc):
    strat = DonchianBreakout(entry_window=10, exit_window=5)
    pos = strat.generate_signals(breakout_ohlc)

    assert pos.loc["2020-01-28"] == 0
    assert pos.loc["2020-01-29"] == 1
    assert pos.iloc[-1] == 1  # stays long, price never falls back below the trailing low


def test_donchian_rejects_exit_window_larger_than_entry():
    with pytest.raises(ValueError):
        DonchianBreakout(entry_window=5, exit_window=10)


def test_buy_and_hold_is_always_fully_long(crossover_ohlc):
    strat = BuyAndHold()
    pos = strat.generate_signals(crossover_ohlc)
    assert (pos == 1).all()


@pytest.mark.parametrize(
    "strategy_cls", [MovingAverageCrossover, RSIMeanReversion, DonchianBreakout, BuyAndHold]
)
def test_param_grid_nonempty_and_constructible(strategy_cls):
    grid = strategy_cls.param_grid()
    assert len(grid) > 0
    for params in grid:
        strategy_cls(**params)  # should not raise


def _random_ohlc(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-04", periods=n, freq="B")
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, n))
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.001, n)),
            "High": close * (1 + np.abs(rng.normal(0, 0.004, n))),
            "Low": close * (1 - np.abs(rng.normal(0, 0.004, n))),
            "Close": close,
            "Volume": 1_000_000,
        },
        index=dates,
    )


@pytest.mark.parametrize(
    "strategy_cls, params",
    [
        (MovingAverageCrossover, {"fast": 5, "slow": 20}),
        (RSIMeanReversion, {"period": 14, "oversold": 30, "overbought": 70}),
        (DonchianBreakout, {"entry_window": 15, "exit_window": 7}),
        (BuyAndHold, {}),
    ],
)
def test_signal_at_time_t_does_not_depend_on_data_after_t(strategy_cls, params):
    """The key correctness property the walk-forward framework relies on: a
    strategy's signal for day t must be a pure function of data through day
    t. If any strategy (or an indicator it calls) ever started peeking ahead
    -- a centered rolling window, a whole-series normalization, a stray
    ``shift(-1)`` -- generating signals on a longer series that shares the
    same history up to a cutoff would change the earlier signal values. This
    guards against that regardless of which strategy it happens in."""
    n = 90
    cutoff = 55
    df = _random_ohlc(n, seed=11)

    strat = strategy_cls(**params)
    signals_full = strat.generate_signals(df)

    # Case 1: simply truncating the series after the cutoff must reproduce
    # identical signals up to the cutoff.
    truncated = df.iloc[:cutoff]
    signals_truncated = strategy_cls(**params).generate_signals(truncated)
    pd.testing.assert_series_equal(
        signals_full.iloc[:cutoff], signals_truncated, check_names=False
    )

    # Case 2: violently perturbing (not just removing) the data after the
    # cutoff must also leave every earlier signal untouched.
    perturbed = df.copy()
    perturbed.iloc[cutoff:] = perturbed.iloc[cutoff:] * 5.0 + 10_000.0
    signals_perturbed = strategy_cls(**params).generate_signals(perturbed)
    pd.testing.assert_series_equal(
        signals_full.iloc[:cutoff], signals_perturbed.iloc[:cutoff], check_names=False
    )
