"""End-to-end demo: loads real daily data, walk-forward tests three
strategies against buy-and-hold on SPY (with charts + trade logs), and runs
a condensed cross-ticker robustness check on AAPL, QQQ and TLT.

Usage: python scripts/run_analysis.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from walkforward_backtester.data import load_prices
from walkforward_backtester.execution import ExecutionConfig, run_backtest
from walkforward_backtester.metrics import summarize
from walkforward_backtester.regime import sharpe_by_regime, trend_regime, volatility_regime
from walkforward_backtester.reporting import (
    plot_drawdowns,
    plot_equity_curves,
    plot_regime_bars,
    plot_trade_pnl_histogram,
)
from walkforward_backtester.strategies import (
    BuyAndHold,
    DonchianBreakout,
    MovingAverageCrossover,
    RSIMeanReversion,
)
from walkforward_backtester.walkforward import (
    WalkForwardSplitter,
    run_walkforward,
    stitch_oos_curve,
    stitch_trade_log,
    walkforward_summary,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
IMG = ROOT / "docs" / "img"
REPORTS.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

TICKERS = ["spy.us", "aapl.us", "qqq.us", "tlt.us"]
PRIMARY = "spy.us"

STRATEGIES = {
    "ma_crossover": MovingAverageCrossover,
    "rsi_mean_reversion": RSIMeanReversion,
    "donchian_breakout": DonchianBreakout,
}

SPLITTER = WalkForwardSplitter(
    train_period=pd.DateOffset(years=2),
    val_period=pd.DateOffset(years=1),
    test_period=pd.DateOffset(months=6),
)
CONFIG = ExecutionConfig(
    initial_capital=100_000.0,
    commission_bps=5.0,
    slippage_bps=5.0,
    position_sizing="fixed_fraction",
    fixed_fraction=0.95,
    max_gross_exposure=1.0,
)


def load_all() -> dict[str, pd.DataFrame]:
    data = {}
    for t in TICKERS:
        df = load_prices(t)
        data[t] = df
        print(f"loaded {t}: {len(df)} rows, {df.index[0].date()} -> {df.index[-1].date()}")
    return data


def buy_and_hold_over_window(df: pd.DataFrame, start, end, config: ExecutionConfig):
    window = df[(df.index >= start) & (df.index < end)]
    signals = BuyAndHold().generate_signals(window)
    return run_backtest(window, signals, config)


def run_ticker(ticker: str, df: pd.DataFrame, make_charts: bool) -> dict:
    print(f"\n=== {ticker} ===")
    outcomes_by_strategy = {}
    curves = {}
    trade_logs = {}
    metrics_rows = []

    for name, cls in STRATEGIES.items():
        t0 = time.time()
        outcomes = run_walkforward(cls, df, SPLITTER, CONFIG)
        outcomes_by_strategy[name] = outcomes
        curve = stitch_oos_curve(outcomes, CONFIG.initial_capital)
        curves[name] = curve
        trades = stitch_trade_log(outcomes)
        trade_logs[name] = trades
        summary = walkforward_summary(outcomes, CONFIG.initial_capital)
        summary["strategy"] = name
        summary["n_folds"] = len(outcomes)
        metrics_rows.append(summary)
        print(f"  {name}: {len(outcomes)} folds, sharpe={summary['sharpe']:.2f}, "
              f"cagr={summary['cagr']:.2%}, maxdd={summary['max_drawdown']:.2%} "
              f"({time.time()-t0:.1f}s)")

    if not outcomes_by_strategy[next(iter(STRATEGIES))]:
        raise RuntimeError(f"No walk-forward folds produced for {ticker}; history too short")

    any_outcomes = next(iter(outcomes_by_strategy.values()))
    oos_start = any_outcomes[0].fold.test_start
    oos_end = any_outcomes[-1].fold.test_end

    bh_result = buy_and_hold_over_window(df, oos_start, oos_end, CONFIG)
    curves["buy_and_hold"] = bh_result.equity_curve
    bh_summary = summarize(bh_result.equity_curve, bh_result.daily_returns, bh_result.trade_log())
    bh_summary["strategy"] = "buy_and_hold"
    bh_summary["n_folds"] = 1
    metrics_rows.append(bh_summary)

    comparison = pd.DataFrame(metrics_rows).set_index("strategy")
    comparison = comparison[
        ["cagr", "sharpe", "max_drawdown", "calmar", "annualized_vol", "num_trades", "win_rate", "profit_factor", "avg_trade_pnl", "n_folds"]
    ]
    comparison.to_csv(REPORTS / f"comparison_{ticker.replace('.', '_')}.csv")
    print(comparison.round(4).to_string())

    # regime analysis, computed on this ticker's own realized vol / trend over the OOS window
    oos_price = df[(df.index >= oos_start) & (df.index < oos_end)]
    vol_regime = volatility_regime(oos_price)
    trnd_regime = trend_regime(oos_price)

    regime_vol_sharpe = {}
    regime_trend_sharpe = {}
    for name, curve in curves.items():
        rets = curve.pct_change().fillna(0.0)
        regime_vol_sharpe[name] = sharpe_by_regime(rets, vol_regime)
        regime_trend_sharpe[name] = sharpe_by_regime(rets, trnd_regime)

    regime_vol_df = pd.DataFrame(regime_vol_sharpe).T
    regime_trend_df = pd.DataFrame(regime_trend_sharpe).T
    regime_vol_df.to_csv(REPORTS / f"regime_volatility_{ticker.replace('.', '_')}.csv")
    regime_trend_df.to_csv(REPORTS / f"regime_trend_{ticker.replace('.', '_')}.csv")

    for name, trades in trade_logs.items():
        trades.to_csv(REPORTS / f"trades_{name}_{ticker.replace('.', '_')}.csv", index=False)

    if make_charts:
        plot_equity_curves(
            curves, f"{ticker.upper()} — walk-forward OOS equity curves", IMG / "equity_curves.svg"
        )
        plot_drawdowns(curves, f"{ticker.upper()} — drawdowns", IMG / "drawdowns.svg")
        best_strategy = comparison.drop(index="buy_and_hold")["sharpe"].idxmax()
        plot_trade_pnl_histogram(
            trade_logs[best_strategy],
            f"{ticker.upper()} — {best_strategy} trade P&L distribution",
            IMG / "trade_pnl_hist.svg",
        )
        plot_regime_bars(
            regime_vol_sharpe, f"{ticker.upper()} — Sharpe by realized-vol regime", IMG / "regime_volatility.svg"
        )
        plot_regime_bars(
            regime_trend_sharpe, f"{ticker.upper()} — Sharpe by trend regime", IMG / "regime_trend.svg"
        )

    return {
        "comparison": comparison,
        "regime_vol": regime_vol_df,
        "regime_trend": regime_trend_df,
        "oos_start": str(oos_start.date()),
        "oos_end": str(oos_end.date()),
        "n_folds": len(any_outcomes),
    }


def main() -> None:
    data = load_all()
    all_results = {}
    for ticker in TICKERS:
        make_charts = ticker == PRIMARY
        all_results[ticker] = run_ticker(ticker, data[ticker], make_charts)

    cross_ticker_rows = []
    for ticker, res in all_results.items():
        comp = res["comparison"]
        for strategy in comp.index:
            row = comp.loc[strategy].to_dict()
            row["ticker"] = ticker
            row["strategy"] = strategy
            cross_ticker_rows.append(row)
    cross_df = pd.DataFrame(cross_ticker_rows).set_index(["ticker", "strategy"])
    cross_df.to_csv(REPORTS / "cross_ticker_summary.csv")

    meta = {ticker: {"oos_start": res["oos_start"], "oos_end": res["oos_end"], "n_folds": res["n_folds"]} for ticker, res in all_results.items()}
    with open(REPORTS / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n=== cross-ticker summary ===")
    print(cross_df.round(4).to_string())
    print("\nDone. Reports written to", REPORTS, "and", IMG)


if __name__ == "__main__":
    main()
