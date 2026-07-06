"""Historical OHLCV data loading with graceful degradation.

Tries stooq.com first (free, keyless, CSV endpoint), then Yahoo Finance's
public chart API as a secondary live source, and finally falls back to a
bundled local cache under ``data/cache/`` so the project always runs even
with no network access at all. Every path returns genuine market data;
nothing here fabricates prices.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[REQUIRED_COLUMNS].dropna()
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    return df


def _fetch_stooq(symbol: str, timeout: float = 8.0) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    text = resp.text.lstrip()
    if text.startswith("<") or "Date,Open" not in text:
        raise ValueError(f"stooq did not return CSV for {symbol} (likely blocked)")
    df = pd.read_csv(io.StringIO(text), index_col="Date")
    return _normalize(df)


def _fetch_yahoo(symbol: str, range_: str = "10y", timeout: float = 8.0) -> pd.DataFrame:
    ticker = symbol.split(".")[0].upper()
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    resp = requests.get(
        url,
        params={"range": range_, "interval": "1d"},
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
    )
    resp.raise_for_status()
    payload = resp.json()
    result = payload["chart"]["result"][0]
    ts = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    idx = pd.to_datetime(ts, unit="s")
    df = pd.DataFrame(
        {
            "Open": quote["open"],
            "High": quote["high"],
            "Low": quote["low"],
            "Close": quote["close"],
            "Volume": quote["volume"],
        },
        index=idx,
    )
    return _normalize(df)


def _load_cache(symbol: str) -> pd.DataFrame:
    path = CACHE_DIR / f"{symbol.lower()}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No cached data bundled for {symbol!r} at {path}")
    df = pd.read_csv(path, index_col="Date")
    return _normalize(df)


def load_prices(
    symbol: str,
    source: str = "auto",
    start: str | None = None,
    end: str | None = None,
    use_cache_fallback: bool = True,
) -> pd.DataFrame:
    """Load daily OHLCV history for ``symbol`` (e.g. ``"spy.us"``).

    ``source`` is one of ``"auto"`` (stooq -> yahoo -> cache), ``"stooq"``,
    ``"yahoo"`` or ``"cache"``. Regardless of the live source reached, the
    result is also written to the local cache so subsequent offline runs are
    reproducible.
    """
    symbol = symbol.lower()
    errors: list[str] = []
    df: pd.DataFrame | None = None
    fetched_live = False

    attempts = {
        "stooq": _fetch_stooq,
        "yahoo": _fetch_yahoo,
    }
    order = [source] if source in attempts else (["stooq", "yahoo"] if source == "auto" else [])

    for name in order:
        try:
            df = attempts[name](symbol)
            if len(df) < 100:
                raise ValueError(f"{name} returned only {len(df)} rows, too little history")
            fetched_live = True
            break
        except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a fallback chain
            errors.append(f"{name}: {exc}")
            df = None

    if df is None:
        if source == "cache" or use_cache_fallback:
            try:
                df = _load_cache(symbol)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"cache: {exc}")
                raise RuntimeError(
                    f"Could not load data for {symbol} from any source: {'; '.join(errors)}"
                ) from exc
        else:
            raise RuntimeError(f"Could not load data for {symbol}: {'; '.join(errors)}")

    if fetched_live:
        try:
            save_to_cache(symbol, df)
        except OSError:
            # Best-effort only -- a read-only filesystem or missing
            # permissions shouldn't fail a load that already succeeded.
            pass

    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    return df


def save_to_cache(symbol: str, df: pd.DataFrame) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{symbol.lower()}.csv"
    df[REQUIRED_COLUMNS].to_csv(path, index_label="Date")
    return path
