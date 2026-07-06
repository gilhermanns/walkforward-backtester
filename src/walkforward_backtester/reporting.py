"""Matplotlib chart helpers, saved as plain-text SVG for portability."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .metrics import drawdown_series

_COLORS = {
    "ma_crossover": "#4C78A8",
    "rsi_mean_reversion": "#F58518",
    "donchian_breakout": "#54A24B",
    "buy_and_hold": "#72727A",
}


def plot_equity_curves(curves: dict[str, pd.Series], title: str, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, curve in curves.items():
        ax.plot(curve.index, curve.values, label=name, color=_COLORS.get(name), linewidth=1.4)
    ax.set_title(title)
    ax.set_ylabel("Equity ($)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    return out_path


def plot_drawdowns(curves: dict[str, pd.Series], title: str, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, curve in curves.items():
        dd = drawdown_series(curve) * 100
        ax.plot(dd.index, dd.values, label=name, color=_COLORS.get(name), linewidth=1.2)
    ax.set_title(title)
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    return out_path


def plot_trade_pnl_histogram(trade_log: pd.DataFrame, title: str, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pnls = trade_log["pnl"].dropna()
    if not pnls.empty:
        ax.hist(pnls, bins=30, color="#4C78A8", edgecolor="white")
    ax.axvline(0, color="#B00", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Trade P&L ($)")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    return out_path


def plot_regime_bars(regime_sharpes: dict[str, pd.Series], title: str, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    strategies = list(regime_sharpes.keys())
    all_regimes = sorted({r for s in regime_sharpes.values() for r in s.index})
    width = 0.8 / max(len(strategies), 1)
    x = range(len(all_regimes))
    for i, name in enumerate(strategies):
        values = [regime_sharpes[name].get(r, 0.0) for r in all_regimes]
        offsets = [xi + i * width for xi in x]
        ax.bar(offsets, values, width=width, label=name, color=_COLORS.get(name))
    ax.set_xticks([xi + width * (len(strategies) - 1) / 2 for xi in x])
    ax.set_xticklabels(all_regimes)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Sharpe ratio")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    return out_path
