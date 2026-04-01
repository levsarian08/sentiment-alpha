"""
End-to-end sentiment alpha pipeline.

Steps:
  1. Fetch news headlines via yfinance
  2. Score with FinBERT
  3. Build daily cross-sectional signal
  4. Fetch price returns
  5. Merge and validate (IC, walk-forward, long/short)
  6. Print summary and save results

Usage
-----
    python run.py
    python run.py --tickers AAPL MSFT TSLA --period 3mo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.news import fetch_news, TICKERS
from src.data.prices import fetch_prices
from src.signal.sentiment import FinBERTScorer
from src.signal.alpha import build_daily_signal, merge_signal_returns
from src.backtest.validate import compute_ic, walk_forward_ic, long_short_returns, hit_rate

RESULTS_DIR = Path("results")


def main(tickers: list[str], period: str) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    # ── 1. Fetch news ─────────────────────────────────────────────────────────
    print("\n[1/5] Fetching news headlines...")
    news = fetch_news(tickers=tickers)
    print(f"  {len(news)} headlines across {news['ticker'].nunique()} tickers")
    if news.empty:
        print("  No headlines found. Exiting.")
        return

    # ── 2. Score with FinBERT ─────────────────────────────────────────────────
    print("\n[2/5] Scoring headlines with FinBERT...")
    scorer = FinBERTScorer()
    scored = scorer.score_dataframe(news)
    scored.to_csv(RESULTS_DIR / "scored_headlines.csv", index=False)
    print(f"  Mean sentiment: {scored['sentiment'].mean():+.4f}")
    print(f"  Most bullish  : {scored.loc[scored['sentiment'].idxmax(), 'headline']}")
    print(f"  Most bearish  : {scored.loc[scored['sentiment'].idxmin(), 'headline']}")

    # ── 3. Build daily signal ─────────────────────────────────────────────────
    print("\n[3/5] Building daily cross-sectional signal...")
    signal = build_daily_signal(scored)
    print(f"  {len(signal)} (ticker, date) observations")
    print(f"  Date range: {signal['date'].min().date()} → {signal['date'].max().date()}")

    # ── 4. Fetch prices ───────────────────────────────────────────────────────
    print("\n[4/5] Fetching price data...")
    prices = fetch_prices(tickers=tickers, period=period)
    print(f"  {len(prices)} price observations")

    # ── 5. Validate ───────────────────────────────────────────────────────────
    print("\n[5/5] Validating signal...")
    merged = merge_signal_returns(signal, prices)

    if len(merged) < 5:
        print("  [!] Not enough overlapping signal/return observations for validation.")
        print("  This can happen when news dates don't overlap with market trading days.")
        print("  Try running with --period 1y for more history.")
        merged.to_csv(RESULTS_DIR / "merged.csv", index=False)
        return

    # Overall IC
    ic_stats = compute_ic(merged)
    hr = hit_rate(merged)

    # Walk-forward IC
    wf = walk_forward_ic(merged, window=max(5, len(merged) // 4))
    wf.to_csv(RESULTS_DIR / "walk_forward_ic.csv", index=False)

    # Long/short
    ls = long_short_returns(merged)
    ls.to_csv(RESULTS_DIR / "long_short.csv", index=False)

    # Save merged
    merged.to_csv(RESULTS_DIR / "merged.csv", index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  SIGNAL VALIDATION SUMMARY")
    print("=" * 50)
    print(f"  Observations    : {ic_stats['n_obs']}")
    print(f"  IC (Spearman)   : {ic_stats['ic']:+.4f}")
    print(f"  IC t-statistic  : {ic_stats['t_stat']:+.4f}")
    print(f"  IC p-value      : {ic_stats['p_value']:.4f}")
    print(f"  Hit rate        : {hr:.1%}")
    if not ls.empty:
        total_ls = ls["cumulative_ls"].iloc[-1]
        print(f"  L/S cum. return : {total_ls:+.1%}")
    print("=" * 50)

    if ic_stats["p_value"] < 0.05:
        print("  ✓ IC is statistically significant (p < 0.05)")
    else:
        print("  ~ IC not significant — more data needed (try --period 1y)")

    print(f"\n  Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=TICKERS)
    parser.add_argument("--period", default="6mo",
                        help="Price history period: 1mo, 3mo, 6mo, 1y, 2y")
    args = parser.parse_args()
    main(args.tickers, args.period)
