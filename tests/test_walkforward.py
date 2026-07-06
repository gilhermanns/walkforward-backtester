import numpy as np
import pandas as pd
import pytest

from walkforward_backtester.execution import ExecutionConfig
from walkforward_backtester.strategies import MovingAverageCrossover
from walkforward_backtester.walkforward import (
    WalkForwardSplitter,
    run_walkforward,
    select_best_params,
    stitch_oos_curve,
)


@pytest.fixture
def long_history() -> pd.DataFrame:
    n = 252 * 6  # 6 years of business days
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0003, 0.01, n)
    prices = 100 * np.cumprod(1 + returns)
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": prices * 1.003,
            "Low": prices * 0.997,
            "Close": prices,
            "Volume": 1_000_000,
        },
        index=dates,
    )
    return df


def test_splitter_produces_non_overlapping_chronological_folds(long_history):
    splitter = WalkForwardSplitter(
        train_period=pd.DateOffset(years=2),
        val_period=pd.DateOffset(years=1),
        test_period=pd.DateOffset(months=6),
    )
    folds = splitter.split(long_history.index)
    assert len(folds) > 0

    for fold in folds:
        assert fold.train_start < fold.train_end == fold.val_start
        assert fold.val_start < fold.val_end == fold.test_start
        assert fold.test_start < fold.test_end

    for prev, nxt in zip(folds, folds[1:]):
        # each fold rolls forward in time; the next fold's train never starts
        # before the previous one's, and folds don't skip backward
        assert nxt.train_start >= prev.train_start
        assert nxt.test_start >= prev.test_start


def test_splitter_windows_have_no_lookahead_overlap(long_history):
    splitter = WalkForwardSplitter(
        train_period=pd.DateOffset(years=2),
        val_period=pd.DateOffset(years=1),
        test_period=pd.DateOffset(months=6),
    )
    folds = splitter.split(long_history.index)
    for fold in folds:
        train_dates = long_history.index[(long_history.index >= fold.train_start) & (long_history.index < fold.train_end)]
        val_dates = long_history.index[(long_history.index >= fold.val_start) & (long_history.index < fold.val_end)]
        test_dates = long_history.index[(long_history.index >= fold.test_start) & (long_history.index < fold.test_end)]

        assert set(train_dates).isdisjoint(set(val_dates))
        assert set(val_dates).isdisjoint(set(test_dates))
        assert set(train_dates).isdisjoint(set(test_dates))
        if len(train_dates) and len(val_dates):
            assert train_dates.max() < val_dates.min()
        if len(val_dates) and len(test_dates):
            assert val_dates.max() < test_dates.min()


def test_splitter_respects_custom_step(long_history):
    splitter = WalkForwardSplitter(
        train_period=pd.DateOffset(years=2),
        val_period=pd.DateOffset(years=1),
        test_period=pd.DateOffset(months=6),
        step=pd.DateOffset(months=3),
    )
    folds = splitter.split(long_history.index)
    assert len(folds) > 1
    gap_days = (folds[1].train_start - folds[0].train_start).days
    assert 85 <= gap_days <= 95  # roughly one quarter, per the custom step


def test_splitter_empty_on_insufficient_history():
    short_index = pd.date_range("2024-01-01", periods=50, freq="B")
    splitter = WalkForwardSplitter(
        train_period=pd.DateOffset(years=2),
        val_period=pd.DateOffset(years=1),
        test_period=pd.DateOffset(months=6),
    )
    assert splitter.split(short_index) == []


def test_run_walkforward_test_segment_never_touches_validation_selection(long_history):
    splitter = WalkForwardSplitter(
        train_period=pd.DateOffset(years=2),
        val_period=pd.DateOffset(years=1),
        test_period=pd.DateOffset(months=6),
    )
    config = ExecutionConfig()
    outcomes = run_walkforward(MovingAverageCrossover, long_history, splitter, config)
    assert len(outcomes) > 0
    for outcome in outcomes:
        assert set(outcome.best_params.keys()) == {"fast", "slow"}
        # the selected params must respect the strategy's own constraint
        assert outcome.best_params["fast"] < outcome.best_params["slow"]
        # test result equity curve must be indexed strictly inside the test window
        idx = outcome.test_result.equity_curve.index
        if len(idx):
            assert idx.min() >= outcome.fold.test_start
            assert idx.max() < outcome.fold.test_end


def test_param_selection_is_blind_to_data_from_the_test_window(long_history):
    """Directly adversarial check: overwrite every bar from the test window
    onward with an explosive, unrelated price path, then confirm the
    validation-selected parameters and score come back byte-for-byte
    identical. If a future edit ever widened the slice used for parameter
    selection (e.g. train_start..test_end instead of train_start..val_end),
    this would fail because the grid search would suddenly "see" a regime
    change that should be invisible to it.
    """
    splitter = WalkForwardSplitter(
        train_period=pd.DateOffset(years=2),
        val_period=pd.DateOffset(years=1),
        test_period=pd.DateOffset(months=6),
    )
    fold = splitter.split(long_history.index)[0]
    config = ExecutionConfig()

    shocked = long_history.copy()
    mask = shocked.index >= fold.test_start
    shock_prices = 100_000 * 1.2 ** np.arange(mask.sum())  # violent, unrelated regime
    shocked.loc[mask, "Open"] = shock_prices
    shocked.loc[mask, "High"] = shock_prices * 1.05
    shocked.loc[mask, "Low"] = shock_prices * 0.95
    shocked.loc[mask, "Close"] = shock_prices

    original_params, original_sharpe = select_best_params(MovingAverageCrossover, long_history, fold, config)
    shocked_params, shocked_sharpe = select_best_params(MovingAverageCrossover, shocked, fold, config)

    assert original_params == shocked_params
    assert original_sharpe == pytest.approx(shocked_sharpe, abs=1e-12)


def test_earlier_fold_outcome_is_unaffected_by_shocking_later_history(long_history):
    """End-to-end version of the same check: run the full walk-forward driver
    twice, differing only in that the second run replaces everything after
    fold 0's own test window with an adversarial price shock. Fold 0's
    selected params, validation score and realized test equity curve must
    come back identical either way -- nothing that happens after a fold's
    test window (including later folds, which run_walkforward also computes
    in the same pass) may reach back and change it.
    """
    splitter = WalkForwardSplitter(
        train_period=pd.DateOffset(years=2),
        val_period=pd.DateOffset(years=1),
        test_period=pd.DateOffset(months=6),
    )
    config = ExecutionConfig()
    baseline = run_walkforward(MovingAverageCrossover, long_history, splitter, config)
    first_test_end = baseline[0].fold.test_end

    shocked = long_history.copy()
    mask = shocked.index >= first_test_end
    rng = np.random.default_rng(99)
    shock_returns = rng.normal(-0.02, 0.06, int(mask.sum()))
    shock_prices = 500 * np.cumprod(1 + shock_returns)
    shocked.loc[mask, "Open"] = shock_prices
    shocked.loc[mask, "High"] = shock_prices * 1.02
    shocked.loc[mask, "Low"] = shock_prices * 0.98
    shocked.loc[mask, "Close"] = shock_prices

    shocked_outcomes = run_walkforward(MovingAverageCrossover, shocked, splitter, config)

    assert baseline[0].best_params == shocked_outcomes[0].best_params
    assert baseline[0].validation_sharpe == pytest.approx(shocked_outcomes[0].validation_sharpe, abs=1e-12)
    pd.testing.assert_series_equal(
        baseline[0].test_result.equity_curve, shocked_outcomes[0].test_result.equity_curve
    )


def test_stitched_oos_curve_is_continuous_and_monotonic_index(long_history):
    splitter = WalkForwardSplitter(
        train_period=pd.DateOffset(years=2),
        val_period=pd.DateOffset(years=1),
        test_period=pd.DateOffset(months=6),
    )
    outcomes = run_walkforward(MovingAverageCrossover, long_history, splitter, ExecutionConfig())
    curve = stitch_oos_curve(outcomes, initial_capital=100_000)
    assert curve.index.is_monotonic_increasing
    assert not curve.index.has_duplicates
    assert curve.iloc[0] > 0
