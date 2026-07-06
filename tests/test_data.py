import pandas as pd
import pytest

from walkforward_backtester import data


def _ohlcv(n=120, start="2020-01-01"):
    dates = pd.date_range(start, periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(n)],
            "High": [100.5 + i for i in range(n)],
            "Low": [99.5 + i for i in range(n)],
            "Close": [100.2 + i for i in range(n)],
            "Volume": [1_000_000] * n,
        },
        index=dates,
    )


def test_normalize_sorts_dedupes_and_drops_bad_rows():
    dates = pd.to_datetime(
        ["2020-01-03", "2020-01-01", "2020-01-02", "2020-01-02"]  # unsorted + duplicate
    )
    df = pd.DataFrame(
        {
            "Open": [10.0, 1.0, 2.0, 2.5],
            "High": [10.5, 1.5, 2.5, -1.0],  # last dup row has a bad (negative) High
            "Low": [9.5, 0.5, 1.5, 2.0],
            "Close": [10.2, 1.2, 2.2, 2.3],
            "Volume": [100, 100, 100, 100],
        },
        index=dates,
    )
    out = data._normalize(df)

    # chronological order, no duplicate index entries
    assert out.index.is_monotonic_increasing
    assert not out.index.duplicated().any()
    # the duplicated 2020-01-02 row had a negative High and keep="last" means
    # it wins the dedupe *before* the positive-price filter runs, so the
    # whole date is correctly dropped rather than silently kept with a bad price
    assert pd.Timestamp("2020-01-02") not in out.index
    assert list(out.index) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-03")]


def test_normalize_strips_timezone_and_time_of_day():
    idx = pd.DatetimeIndex(["2020-01-01 09:30", "2020-01-02 16:00"]).tz_localize("America/New_York")
    df = pd.DataFrame(
        {"Open": [1.0, 2.0], "High": [1.1, 2.1], "Low": [0.9, 1.9], "Close": [1.05, 2.05], "Volume": [10, 20]},
        index=idx,
    )
    out = data._normalize(df)
    assert out.index.tz is None
    assert list(out.index) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]


def test_load_prices_auto_falls_back_from_stooq_to_yahoo(monkeypatch):
    calls = []

    def fail_stooq(symbol, timeout=8.0):
        calls.append("stooq")
        raise ValueError("stooq did not return CSV for spy.us (likely blocked)")

    def ok_yahoo(symbol, range_="10y", timeout=8.0):
        calls.append("yahoo")
        return _ohlcv()

    monkeypatch.setattr(data, "_fetch_stooq", fail_stooq)
    monkeypatch.setattr(data, "_fetch_yahoo", ok_yahoo)
    monkeypatch.setattr(data, "save_to_cache", lambda symbol, df: None)

    df = data.load_prices("spy.us")
    assert calls == ["stooq", "yahoo"]
    assert len(df) == 120


def test_load_prices_falls_back_to_cache_when_all_live_sources_fail(monkeypatch, tmp_path):
    def fail(symbol, **kwargs):
        raise ConnectionError("network unreachable")

    monkeypatch.setattr(data, "_fetch_stooq", fail)
    monkeypatch.setattr(data, "_fetch_yahoo", fail)
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)

    cached = _ohlcv(n=150)
    cached.to_csv(tmp_path / "spy.us.csv", index_label="Date")

    df = data.load_prices("spy.us")
    assert len(df) == 150


def test_load_prices_raises_when_every_source_fails(monkeypatch, tmp_path):
    def fail(symbol, **kwargs):
        raise ConnectionError("network unreachable")

    monkeypatch.setattr(data, "_fetch_stooq", fail)
    monkeypatch.setattr(data, "_fetch_yahoo", fail)
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)  # empty dir, no bundled cache file

    with pytest.raises(RuntimeError):
        data.load_prices("spy.us")


def test_load_prices_writes_successful_live_fetch_to_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "_fetch_stooq", lambda symbol, **kwargs: _ohlcv())
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)

    data.load_prices("spy.us", source="stooq")

    cache_file = tmp_path / "spy.us.csv"
    assert cache_file.exists()
    reloaded = pd.read_csv(cache_file, index_col="Date")
    assert len(reloaded) == 120


def test_load_prices_cache_only_source_never_calls_live_fetchers(monkeypatch, tmp_path):
    def unexpected(symbol, **kwargs):
        raise AssertionError("cache-only load should never hit a live source")

    monkeypatch.setattr(data, "_fetch_stooq", unexpected)
    monkeypatch.setattr(data, "_fetch_yahoo", unexpected)
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    _ohlcv(n=110).to_csv(tmp_path / "spy.us.csv", index_label="Date")

    df = data.load_prices("spy.us", source="cache")
    assert len(df) == 110


def test_load_prices_applies_start_end_filters(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "_fetch_stooq", lambda symbol, **kwargs: _ohlcv(n=120))
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)

    df = data.load_prices("spy.us", source="stooq", start="2020-02-01", end="2020-03-01")
    assert df.index.min() >= pd.Timestamp("2020-02-01")
    assert df.index.max() <= pd.Timestamp("2020-03-01")


def test_load_prices_rejects_thin_history(monkeypatch, tmp_path):
    # A live source returning fewer than 100 rows should be treated as a
    # failed attempt (e.g. a truncated or rate-limited response), not silently
    # accepted as "the" history for the symbol.
    monkeypatch.setattr(data, "_fetch_stooq", lambda symbol, **kwargs: _ohlcv(n=10))
    monkeypatch.setattr(data, "_fetch_yahoo", lambda symbol, **kwargs: _ohlcv(n=120))
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)

    df = data.load_prices("spy.us")
    assert len(df) == 120
