"""
Streamlit dashboard — FinBERT Sentiment Alpha Explorer.

Launch with:
    streamlit run app/dashboard.py

Loads pre-computed results from results/ folder.
Click 'Run pipeline' to recompute from scratch.
"""

from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Sentiment Alpha Explorer",
    page_icon="📈",
    layout="wide",
)

st.title("📈 FinBERT Sentiment Alpha Explorer")
st.caption("Financial news sentiment signal vs. next-day returns — AAPL, GOOGL, MSFT, AMZN, TSLA, META")

from src.backtest.validate import compute_ic, walk_forward_ic, long_short_returns, hit_rate, plot_results

RESULTS = Path("results")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    run_btn = st.button("Run pipeline", type="primary")
    st.markdown("---")
    st.caption("Results are cached in results/. Click 'Run pipeline' to recompute.")

# ── Run pipeline if requested ─────────────────────────────────────────────────
if run_btn:
    from src.data.news import fetch_news, TICKERS
    from src.data.prices import fetch_prices
    from src.signal.sentiment import FinBERTScorer
    from src.signal.alpha import build_daily_signal, merge_signal_returns

    with st.spinner("Fetching news..."):
        news = fetch_news(tickers=TICKERS)

    if news.empty:
        st.error("No headlines found.")
        st.stop()

    with st.spinner("Scoring with FinBERT..."):
        scorer = FinBERTScorer()
        scored = scorer.score_dataframe(news)
        RESULTS.mkdir(exist_ok=True)
        scored.to_csv(RESULTS / "scored_headlines.csv", index=False)

    signal = build_daily_signal(scored)
    prices = fetch_prices(tickers=TICKERS, period="max")
    merged = merge_signal_returns(signal, prices)
    merged.to_csv(RESULTS / "merged.csv", index=False)
    st.success("Pipeline complete! Results saved.")

# ── Load results ──────────────────────────────────────────────────────────────
merged_path = RESULTS / "merged.csv"
headlines_path = RESULTS / "scored_headlines.csv"

if not merged_path.exists():
    st.info("No results found. Click **Run pipeline** in the sidebar to get started, or run `python run.py --period max` in your terminal first.")
    st.stop()

merged = pd.read_csv(merged_path, parse_dates=["date"])
scored = pd.read_csv(headlines_path) if headlines_path.exists() else pd.DataFrame()

if len(merged) < 3:
    st.warning("Not enough data in results. Run `python run.py --period max` in your terminal first.")
    st.stop()

# ── Metrics ───────────────────────────────────────────────────────────────────
ic_stats = compute_ic(merged)
hr = hit_rate(merged)
ls = long_short_returns(merged)
total_ls = ls["cumulative_ls"].iloc[-1] if not ls.empty else float("nan")

col1, col2, col3, col4 = st.columns(4)
col1.metric("IC (Spearman)", f"{ic_stats['ic']:+.4f}")
col2.metric("IC t-statistic", f"{ic_stats['t_stat']:+.3f}")
col3.metric("Hit rate", f"{hr:.1%}")
col4.metric("L/S cumulative return", f"{total_ls:+.1%}" if not np.isnan(total_ls) else "N/A")

st.caption(f"Based on {ic_stats['n_obs']} (ticker, date) observations · "
           f"Date range: {merged['date'].min().date()} → {merged['date'].max().date()}")

# ── Main chart ────────────────────────────────────────────────────────────────
wf = walk_forward_ic(merged, window=max(5, len(merged) // 6))
fig = plot_results(merged, wf, ls)
st.plotly_chart(fig, use_container_width=True)

# ── Headlines table ───────────────────────────────────────────────────────────
if not scored.empty:
    st.subheader("Scored headlines sample")
    display_cols = ["ticker", "date", "headline", "sentiment", "prob_positive", "prob_negative"]
    available = [c for c in display_cols if c in scored.columns]
    top = scored[available].sort_values("sentiment", ascending=False)
    st.markdown("**Most bullish**")
    st.dataframe(top.head(10), use_container_width=True)
    st.markdown("**Most bearish**")
    st.dataframe(top.tail(10), use_container_width=True)
