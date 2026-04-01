# FinBERT Sentiment Alpha

Extracting a quantitative alpha signal from financial news headlines using **FinBERT** and validating its predictive power on next-day returns via **walk-forward Information Coefficient (IC) analysis**.

Applies the core quant research workflow — signal construction from alternative data, rigorous out-of-sample validation, and long/short portfolio simulation — to six large-cap tech stocks using real-world news data.

---

## Results

| Metric | Value |
|---|---|
| IC (Spearman) | ~+0.08 |
| IC t-statistic | ~1.9 |
| Hit rate | ~54% |
| L/S cumulative return | varies by period |

> IC > 0.05 is considered meaningful in quantitative research. Results vary with news volume and market conditions — run `python run.py --period 1y` for more stable estimates.

---

## Project Structure

```
sentiment-alpha/
├── src/
│   ├── data/
│   │   ├── news.py          # yfinance headline fetcher
│   │   └── prices.py        # price data + forward return construction
│   ├── signal/
│   │   ├── sentiment.py     # FinBERT scorer (ProsusAI/finbert)
│   │   └── alpha.py         # daily signal aggregation + cross-sectional z-score
│   └── backtest/
│       └── validate.py      # IC, walk-forward, long/short portfolio, Plotly charts
├── app/
│   └── dashboard.py         # Streamlit signal explorer
├── notebooks/
│   └── research.ipynb       # EDA + results walkthrough
├── run.py                   # end-to-end pipeline script
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/levsarian08/sentiment-alpha.git
cd sentiment-alpha
pip install -r requirements.txt

# 2. Run the full pipeline
python run.py

# 3. Run with custom tickers and longer history
python run.py --tickers AAPL MSFT TSLA --period 1y

# 4. Launch the interactive dashboard
streamlit run app/dashboard.py
```

---

## Methodology

### 1. Data collection
Financial news headlines are fetched for each ticker via yfinance. Each headline carries a publication timestamp, enabling alignment with next-day trading returns.

### 2. Sentiment scoring (FinBERT)
Headlines are scored using [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert), a BERT model fine-tuned on financial news corpora. The continuous sentiment score is:

```
score = P(positive) - P(negative)  ∈ [-1, 1]
```

### 3. Signal construction
Headline-level scores are aggregated to a daily (ticker, date) signal by averaging. The signal is then **cross-sectionally z-scored** each day — removing the market-wide sentiment level to produce a relative ranking signal: long the most positive tickers, short the most negative.

### 4. Walk-forward validation
IC (Spearman rank correlation between signal and next-day returns) is computed in rolling windows to test consistency over time and avoid lookahead bias. This mirrors the out-of-sample validation protocol used in production quant research.

### 5. Long/short simulation
Each day, the top tercile of signal is placed in a simulated long position and the bottom tercile in short, producing a daily L/S return series and cumulative equity curve.

---

## Tech Stack

`Python` · `PyTorch` · `Transformers (HuggingFace)` · `yfinance` · `Pandas` · `NumPy` · `SciPy` · `Streamlit` · `Plotly`
