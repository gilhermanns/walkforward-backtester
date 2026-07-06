import numpy as np
import pandas as pd
import pytest

from walkforward_backtester.execution import ExecutionConfig, _slippage_bps, run_backtest


def test_next_bar_fill_and_commission_slippage_math(toy_trade_df):
    target = pd.Series([1, 1, 0, 0, 0], index=toy_trade_df.index)
    config = ExecutionConfig(initial_capital=100_000, commission_bps=5, slippage_bps=5, fixed_fraction=0.95)
    result = run_backtest(toy_trade_df, target, config)
    log = result.trade_log()

    assert len(log) == 1
    trade = log.iloc[0]

    # Entered at day-2's open (100), one day after the day-1 signal fired.
    assert trade["entry_date"] == toy_trade_df.index[1]
    # Buy-side slippage pushes the fill price up by 5 bps.
    assert trade["entry_price"] == pytest.approx(100.0 * 1.0005, abs=1e-9)
    # Fixed-fraction sizing: 95% of 100,000 capital at a $100 entry price.
    assert trade["shares"] == pytest.approx(950.0, abs=1e-6)

    # Exited at day-4's open (110), one day after the day-3 flat signal.
    assert trade["exit_date"] == toy_trade_df.index[3]
    assert trade["exit_price"] == pytest.approx(110.0 * 0.9995, abs=1e-9)

    expected_commission = abs(trade["entry_price"] * trade["shares"]) * 5 / 10_000 + abs(
        trade["exit_price"] * trade["shares"]
    ) * 5 / 10_000
    assert trade["commission_paid"] == pytest.approx(expected_commission, rel=1e-9)

    expected_slippage = abs(100.0 - trade["entry_price"]) * trade["shares"] + abs(
        110.0 - trade["exit_price"]
    ) * trade["shares"]
    assert trade["slippage_paid"] == pytest.approx(expected_slippage, rel=1e-9)

    expected_pnl = (trade["exit_price"] - trade["entry_price"]) * trade["shares"] - expected_commission - expected_slippage
    assert trade["pnl"] == pytest.approx(expected_pnl, rel=1e-9)
    assert trade["pnl"] > 0


def test_flat_signal_produces_no_trades(toy_trade_df):
    target = pd.Series(0, index=toy_trade_df.index)
    result = run_backtest(toy_trade_df, target, ExecutionConfig())
    assert result.trade_log().empty
    assert (result.equity_curve == result.equity_curve.iloc[0]).all()


def test_stop_loss_exits_before_signal_and_caps_loss():
    dates = pd.date_range("2021-01-04", periods=5, freq="B")
    df = pd.DataFrame(
        {
            "Open": [100.0, 100.0, 110.0, 110.0, 110.0],
            "High": [100.0, 105.0, 111.0, 110.0, 110.0],
            "Low": [100.0, 99.0, 90.0, 110.0, 110.0],
            "Close": [100.0, 104.0, 95.0, 110.0, 110.0],
            "Volume": [1_000_000] * 5,
        },
        index=dates,
    )
    target = pd.Series(1, index=dates)  # strategy wants to stay long throughout
    config = ExecutionConfig(
        initial_capital=100_000, commission_bps=5, slippage_bps=5, fixed_fraction=0.95, stop_loss_pct=0.05
    )
    result = run_backtest(df, target, config)
    log = result.trade_log()

    first_trade = log.iloc[0]
    assert first_trade["exit_reason"] == "stop_loss"
    # Stop fires when the low breaches entry_price * (1 - 5%); fill should sit
    # right around that level (within the slippage adjustment).
    stop_price = first_trade["entry_price"] * 0.95
    assert first_trade["exit_price"] == pytest.approx(stop_price, rel=5e-3)
    assert first_trade["pnl"] < 0


def test_take_profit_exits_when_price_spikes():
    dates = pd.date_range("2021-01-04", periods=5, freq="B")
    df = pd.DataFrame(
        {
            "Open": [100.0, 100.0, 110.0, 110.0, 110.0],
            "High": [100.0, 130.0, 111.0, 110.0, 110.0],
            "Low": [100.0, 99.0, 109.0, 110.0, 110.0],
            "Close": [100.0, 104.0, 110.0, 110.0, 110.0],
            "Volume": [1_000_000] * 5,
        },
        index=dates,
    )
    target = pd.Series(1, index=dates)
    config = ExecutionConfig(
        initial_capital=100_000, commission_bps=5, slippage_bps=5, fixed_fraction=0.95, take_profit_pct=0.10
    )
    result = run_backtest(df, target, config)
    log = result.trade_log()
    assert log.iloc[0]["exit_reason"] == "take_profit"
    assert log.iloc[0]["pnl"] > 0


def test_vol_scaled_slippage_baseline_ignores_future_bars():
    # Regression test: the vol-scaled slippage model's "baseline" reference
    # volatility used to be computed from a fixed early window of the frame
    # regardless of the current bar, so for a bar near the start of the
    # series that baseline window could reach past the bar itself into data
    # that wouldn't yet exist at that point in time. Perturbing prices well
    # after an early bar must not change the slippage bps computed at that
    # bar.
    dates = pd.date_range("2021-01-04", periods=40, freq="B")
    rng = np.random.default_rng(3)
    prices = 100 * np.cumprod(1 + rng.normal(0.0, 0.005, 40))
    df = pd.DataFrame(
        {"Open": prices, "High": prices, "Low": prices, "Close": prices, "Volume": [1_000_000] * 40},
        index=dates,
    )
    config = ExecutionConfig(slippage_model="vol_scaled", vol_scale_lookback=20, slippage_bps=5.0)

    early_bar = 5
    bps_before = _slippage_bps(config, df, early_bar)

    # Perturb bars strictly after `early_bar` but still inside the first
    # `vol_scale_lookback` (20) bars of the frame -- i.e. bars 5..19 are the
    # future relative to bar 5, yet a fixed "first 20 bars" baseline window
    # would still reach into them.
    mutated = df.copy()
    mutated.loc[mutated.index[10:20], "Close"] = mutated["Close"].iloc[10:20] * 4.0 + 1_000.0

    bps_after = _slippage_bps(config, mutated, early_bar)
    assert bps_after == pytest.approx(bps_before)


def test_max_gross_exposure_caps_vol_target_sizing():
    dates = pd.date_range("2021-01-04", periods=30, freq="B")
    # small daily oscillation gives a low but non-zero realized volatility
    prices = [100.0 + (0.3 if i % 2 == 0 else -0.3) for i in range(30)]
    df = pd.DataFrame(
        {"Open": prices, "High": prices, "Low": prices, "Close": prices, "Volume": [1_000_000] * 30},
        index=dates,
    )
    target = pd.Series(1, index=dates)
    config = ExecutionConfig(
        initial_capital=100_000,
        position_sizing="vol_target",
        vol_target_annual=10.0,  # deliberately huge, to try to force leverage well above 1x
        max_gross_exposure=1.0,
    )
    result = run_backtest(df, target, config)
    notional = (result.positions * df["Close"]).abs()
    # even with an aggressive vol target, notional exposure should never
    # exceed 1x capital because of the max_gross_exposure cap
    assert (notional <= 100_000 * 1.0001).all()
    assert notional.max() > 0  # sanity: a position was actually taken
