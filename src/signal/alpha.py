"""
Construct a daily sentiment alpha signal from scored headlines.

Pipeline:
  1. Average sentiment scores across all headlines for a given (ticker, date)
  2. Cross-sectionally z-score the signal each day (removes market-wide bias)
  3. Merge with forward returns for validation

Cross-sectional z-scoring is standard quant practice:
  - Removes the overall market sentiment level each day
  - Makes the signal a relative bet: long the most positive, short the most negative
  - Reduces correlation with broad market moves
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_daily_signal(scored_news: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate headline-level sentiment scores to a daily (ticker, date) signal.

    Parameters
    ----------
    scored_news : DataFrame with columns ticker, date, sentiment (from FinBERT)

    Returns
    -------
    DataFrame with columns: ticker, date, raw_sentiment, signal, n_headlines
    where `signal` is the cross-sectionally z-scored sentiment.
    """
    scored_news = scored_news.copy()
    scored_news["date"] = pd.to_datetime(scored_news["date"]).dt.normalize()

    # Aggregate: mean sentiment per (ticker, date), count headlines
    daily = (
        scored_news
        .groupby(["ticker", "date"])
        .agg(
            raw_sentiment=("sentiment", "mean"),
            n_headlines=("sentiment", "count"),
        )
        .reset_index()
    )

    # Cross-sectional z-score: subtract daily mean, divide by daily std
    def _zscore(x):
        mu, sigma = x.mean(), x.std()
        return (x - mu) / (sigma + 1e-9)

    daily["signal"] = (
        daily.groupby("date")["raw_sentiment"]
        .transform(_zscore)
    )

    return daily.sort_values(["date", "ticker"]).reset_index(drop=True)


def merge_signal_returns(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge daily signal with forward returns on (ticker, date).

    Parameters
    ----------
    signal : output of build_daily_signal()
    prices : output of fetch_prices() with columns ticker, date, return_1d

    Returns
    -------
    Merged DataFrame with signal and return_1d aligned by (ticker, date).
    Rows with missing returns or signal are dropped.
    """
    prices = prices[["ticker", "date", "return_1d"]].copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()

    merged = pd.merge(signal, prices, on=["ticker", "date"], how="inner")
    merged = merged.dropna(subset=["signal", "return_1d"])
    return merged.sort_values(["date", "ticker"]).reset_index(drop=True)
