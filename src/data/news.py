"""
Fetch financial news headlines from the HuggingFace financial-news-articles dataset.
Dates are extracted from URLs using regex. Dataset covers ~2017-2022.
"""

from __future__ import annotations

import re
import pandas as pd
from tqdm import tqdm

try:
    from datasets import load_dataset
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "-q"])
    from datasets import load_dataset

TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META"]

TICKER_KEYWORDS = {
    "AAPL":  ["apple", "aapl", "iphone", "tim cook", "macbook", "ios"],
    "GOOGL": ["google", "googl", "alphabet", "sundar pichai", "youtube"],
    "MSFT":  ["microsoft", "msft", "satya nadella", "azure", "windows"],
    "AMZN":  ["amazon", "amzn", "aws", "andy jassy", "jeff bezos", "prime"],
    "TSLA":  ["tesla", "tsla", "elon musk", "electric vehicle"],
    "META":  ["meta", "facebook", "instagram", "zuckerberg", "whatsapp"],
}

DATE_PATTERNS = [
    r"/(\d{4})/(\d{2})/(\d{2})/",
    r"-(\d{4})-(\d{2})-(\d{2})-",
    r"(\d{4})(\d{2})(\d{2})",
]


def _extract_date_from_url(url: str) -> pd.Timestamp | None:
    for pattern in DATE_PATTERNS:
        m = re.search(pattern, str(url))
        if m:
            try:
                year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if 2010 <= year <= 2026 and 1 <= month <= 12 and 1 <= day <= 31:
                    return pd.Timestamp(year=year, month=month, day=day)
            except Exception:
                continue
    return None


def _matches_ticker(headline: str, ticker: str) -> bool:
    text = headline.lower()
    return any(kw in text for kw in TICKER_KEYWORDS.get(ticker, [ticker.lower()]))


def fetch_news(
    tickers: list[str] = TICKERS,
    max_per_ticker: int = 300,
    lookback_days: int = None,   # None = use all available data
) -> pd.DataFrame:
    print("  Loading dataset from HuggingFace (cached after first run)...")
    dataset = load_dataset("ashraq/financial-news-articles", split="train")
    df_all = dataset.to_pandas()
    print(f"  {len(df_all):,} total headlines loaded")

    print("  Extracting dates from URLs...")
    df_all["date"] = df_all["url"].apply(_extract_date_from_url)
    df_all = df_all.dropna(subset=["date"])
    print(f"  {len(df_all):,} headlines with extractable dates")
    print(f"  Dataset date range: {df_all['date'].min().date()} → {df_all['date'].max().date()}")

    if lookback_days is not None:
        cutoff = df_all["date"].max() - pd.Timedelta(days=lookback_days)
        df_all = df_all[df_all["date"] >= cutoff]
        print(f"  {len(df_all):,} headlines after lookback filter")

    if df_all.empty:
        return pd.DataFrame()

    rows = []
    for ticker in tqdm(tickers, desc="Filtering by ticker"):
        mask = df_all["title"].astype(str).str.lower().apply(
            lambda h: _matches_ticker(h, ticker)
        )
        matched = df_all[mask].sort_values("date", ascending=False).head(max_per_ticker)
        for _, row in matched.iterrows():
            rows.append({
                "ticker":    ticker,
                "date":      row["date"],
                "headline":  str(row["title"]),
                "publisher": "",
                "url":       str(row.get("url", "")),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.drop_duplicates(subset=["ticker", "headline"])
    df = df.sort_values("date", ascending=False).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = fetch_news()
    if not df.empty:
        print(f"\nFetched {len(df)} headlines across {df['ticker'].nunique()} tickers")
        print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
        print(df[["ticker", "date", "headline"]].head(10).to_string(index=False))
