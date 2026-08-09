<h1>Walk-Forward Strategy Backtester</h1>

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A robust, event-driven backtesting engine designed for **rigorous validation of quantitative trading strategies** using a walk-forward optimization framework. This project is critical for **Wealth Management, Private Banking, and Institutional Asset Management** professionals conducting due diligence on investment products, managing risk, and evaluating systematic strategies.

<h2>Business Relevance</h2>

For financial professionals, particularly those in **Wealth Management, Private Banking, and Institutional Asset Management**, the ability to thoroughly vet investment strategies is paramount. This backtester provides a framework for:

-   **Due Diligence**: Objectively assessing the true out-of-sample performance of systematic trading strategies before recommending them to clients or integrating them into portfolios.
-   **Risk Management**: Understanding the robustness of strategies across different market regimes and identifying potential vulnerabilities to overfitting.
-   **Product Development**: Building and refining quantitative investment products with confidence, knowing that performance metrics are derived from genuinely out-of-sample data.
-   **Client Communication**: Providing transparent and credible performance analytics to clients, demonstrating a rigorous approach to strategy selection and management.

This project emphasizes the importance of realistic execution frictions and proper validation techniques to distinguish genuinely robust strategies from those that merely appear successful due to data snooping or overfitting.

<h2>Motivation (Applied Perspective)</h2>

Many quantitative strategies, especially those based on technical indicators, are prone to **overfitting** – performing well on historical data but failing in live markets. This project directly addresses this challenge by asking a more honest and practical question:

**When strategy parameters are selected exclusively on historical data and then tested strictly on unseen data, do common rule-based strategies (e.g., moving-average crossover, RSI mean-reversion, Donchian breakout) deliver consistent out-of-sample performance? Furthermore, how does their efficacy vary across different market regimes?**

By employing a rolling walk-forward validation framework and incorporating realistic execution costs (commission, slippage, next-bar fills, stop-loss/take-profit, position sizing), this backtester ensures that every reported performance metric is genuinely out-of-sample and reflective of real-world trading conditions.

<h2>Methodology</h2>

-   **Strategies (`src/walkforward_backtester/strategies.py`)**: Implements classic rule-based strategies: Moving Average Crossover, RSI Mean-Reversion, and Donchian Breakout. Each strategy adheres to a `generate_signals(df) -> Series` interface for consistent evaluation.
-   **Execution (`execution.py`)**: A bar-by-bar simulation that ensures signals generated from day *t* are actionable only at the open of day *t+1*, preventing look-ahead bias. It accounts for realistic execution frictions such as commissions, slippage (fixed or volatility-scaled), and position sizing (fixed fraction or volatility-targeted). Optional percentage stop-loss and take-profit levels are checked intrabar.
-   **Walk-Forward Validation (`walkforward.py`)**: Divides historical data into rolling folds (e.g., **2 years training → 1 year validation → 6 months testing**), advancing by 6 months each time. Parameters are optimized solely on the validation segment (e.g., maximizing Sharpe ratio), and the winning parameters are then applied, untouched, to the subsequent out-of-sample test segment. This methodology ensures all reported results are genuinely out-of-sample.
-   **Regime Analysis (`regime.py`)**: Post-hoc analysis of strategy performance across different market regimes (e.g., realized-volatility terciles, trend vs. sideways markets) to provide descriptive insights into strategy behavior under varying conditions.

<h2>Sample Output & Insights</h2>

This backtester produces comprehensive performance metrics and visualizations, crucial for strategy evaluation and client reporting:

-   **Headline Comparison**: Detailed performance statistics (CAGR, Sharpe, Max Drawdown, Calmar, Annualized Volatility, Number of Trades, Win Rate, Profit Factor, Average Trade P&L) for each strategy against a buy-and-hold benchmark.
-   **Equity Curves & Drawdowns**: Visual representations of cumulative growth and drawdowns, illustrating the path of capital over time.
-   **Trade P&L Distribution**: Histograms showing the distribution of individual trade profits and losses, providing insight into strategy consistency.
-   **Regime Split Analysis**: Performance metrics broken down by market regime (e.g., high/mid/low volatility, trending/sideways markets), revealing how strategies adapt to different market environments.
-   **Cross-Ticker Robustness**: Comparison of strategy performance across various assets (e.g., SPY, AAPL, QQQ, TLT) to assess generalizability.

*(Example charts and tables would be embedded here, showcasing equity curves, drawdown charts, trade P&L histograms, and regime-based performance tables.)*

<h2>Project Structure</h2>

```text
/walkforward-backtester
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── scripts/                # Utility scripts for running backtests and analysis
└── src/
    ├── walkforward_backtester/ # Core backtesting logic
    │   ├── data.py         # Data fetching and caching
    │   ├── strategies.py   # Trading strategy implementations
    │   ├── execution.py    # Trade execution and simulation
    │   ├── walkforward.py  # Walk-forward validation framework
    │   └── regime.py       # Market regime analysis
├── tests/                  # Pytest suite
└── reports/                # Generated performance reports and CSVs
```

<h2>Getting Started</h2>

<h3>Installation</h3>
1. Clone the repository:
   ```bash
   git clone https://github.com/gilhermanns/walkforward-backtester.git
   cd walkforward-backtester
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the full analysis:
   ```bash
   python scripts/run_analysis.py
   ```

<h2>License & Disclaimer</h2>

This project is licensed under the MIT License. It is intended for educational and research purposes in quantitative finance. The models and results presented are for illustrative purposes and do not constitute financial advice or guarantee real-world performance. Always exercise professional judgment and conduct thorough due diligence when evaluating investment strategies.

---

*Entwickelt mit Unterstützung von Claude Code (Anthropic).*
