"""
Fetch historical price data and compute forward returns via yfinance.

Forward returns are the key dependent variable in signal validation:
we test whether today's sentiment score predicts tomorrow's return.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META"]


def fetch_prices(
    tickers: list[str] = TICKERS,
    period: str = "6mo",
) -> pd.DataFrame:
    """
    Download adjusted close prices and compute 1-day forward returns.

    Parameters
    ----------
    tickers : list of ticker symbols
    period  : yfinance period string (e.g. '6mo', '1y', '2y')

    Returns
    -------
    DataFrame with columns: ticker, date, close, return_1d
    where return_1d is the NEXT day's return (forward-looking).
    """
    rows = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if df.empty:
                continue

            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            prices = df[["Close"]].copy()
            prices.columns = ["close"]
            prices.index = pd.to_datetime(prices.index).tz_localize(None)

            # Forward return: what you earn by holding tomorrow
            prices["return_1d"] = prices["close"].pct_change().shift(-1)
            prices["ticker"] = ticker
            prices["date"] = prices.index.normalize()
            rows.append(prices.reset_index(drop=True))
        except Exception as e:
            print(f"  [!] {ticker}: {e}")

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)[["ticker", "date", "close", "return_1d"]]


if __name__ == "__main__":
    df = fetch_prices()
    print(df.groupby("ticker").tail(3).to_string(index=False))
