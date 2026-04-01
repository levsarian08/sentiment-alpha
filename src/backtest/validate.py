"""
Signal validation via Information Coefficient (IC) and walk-forward analysis.

Information Coefficient (IC):
    Spearman rank correlation between the signal and forward returns.
    IC > 0 means the signal has positive predictive power.
    IC > 0.05 is considered meaningful in practice.
    IC > 0.10 is considered strong.

Walk-forward validation:
    Roll a window through time, computing IC at each step.
    This tests whether the signal is consistently predictive
    and avoids lookahead bias — the core methodological requirement
    for any credible quant signal research.

We also compute:
    - IC t-statistic (is IC significantly different from zero?)
    - Long/short portfolio returns (top vs bottom signal quintile)
    - Hit rate (% of days where signal direction was correct)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


# ── Information Coefficient ───────────────────────────────────────────────────

def compute_ic(df: pd.DataFrame) -> dict:
    """
    Compute Spearman IC between signal and next-day returns.

    Parameters
    ----------
    df : DataFrame with columns signal, return_1d

    Returns
    -------
    dict with ic, t_stat, p_value, n_obs
    """
    clean = df.dropna(subset=["signal", "return_1d"])
    if len(clean) < 5:
        return {"ic": float("nan"), "t_stat": float("nan"), "p_value": float("nan"), "n_obs": len(clean)}

    ic, p_value = stats.spearmanr(clean["signal"], clean["return_1d"])
    n = len(clean)
    t_stat = ic * np.sqrt((n - 2) / (1 - ic**2 + 1e-9))

    return {"ic": ic, "t_stat": t_stat, "p_value": p_value, "n_obs": n}


def walk_forward_ic(
    df: pd.DataFrame,
    window: int = 10,
) -> pd.DataFrame:
    """
    Compute rolling IC across dates using a walk-forward window.

    Parameters
    ----------
    df     : merged signal + returns DataFrame with a `date` column
    window : number of trading days per IC calculation window

    Returns
    -------
    DataFrame with columns: date, ic, n_obs
    """
    dates = sorted(df["date"].unique())
    rows = []

    for i in range(window, len(dates)):
        window_dates = dates[i - window : i]
        subset = df[df["date"].isin(window_dates)]
        m = compute_ic(subset)
        rows.append({"date": dates[i], **m})

    return pd.DataFrame(rows)


# ── Long/short portfolio ──────────────────────────────────────────────────────

def long_short_returns(
    df: pd.DataFrame,
    quantile: float = 0.33,
) -> pd.DataFrame:
    """
    Simulate a daily long/short portfolio:
      - Long the top `quantile` of signal each day
      - Short the bottom `quantile` of signal each day

    Parameters
    ----------
    df       : merged signal + returns DataFrame
    quantile : fraction of tickers in long and short legs

    Returns
    -------
    DataFrame with columns: date, long_ret, short_ret, ls_ret (long - short)
    """
    rows = []
    for date, group in df.groupby("date"):
        if len(group) < 3:
            continue
        lo = group["signal"].quantile(quantile)
        hi = group["signal"].quantile(1 - quantile)
        longs  = group[group["signal"] >= hi]["return_1d"].mean()
        shorts = group[group["signal"] <= lo]["return_1d"].mean()
        rows.append({
            "date":      date,
            "long_ret":  longs,
            "short_ret": shorts,
            "ls_ret":    longs - shorts,
        })

    result = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    result["cumulative_ls"] = (1 + result["ls_ret"]).cumprod() - 1
    return result


def hit_rate(df: pd.DataFrame) -> float:
    """Fraction of observations where signal direction matches return direction."""
    clean = df.dropna(subset=["signal", "return_1d"])
    correct = (np.sign(clean["signal"]) == np.sign(clean["return_1d"])).mean()
    return float(correct)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(
    df: pd.DataFrame,
    wf_ic: pd.DataFrame,
    ls: pd.DataFrame,
) -> go.Figure:
    """
    Four-panel Plotly dashboard:
      1. Signal vs next-day return scatter (all observations)
      2. Walk-forward IC over time
      3. Cumulative long/short portfolio return
      4. Per-ticker IC bar chart
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Signal vs next-day return",
            "Walk-forward IC (rolling window)",
            "Cumulative long/short return",
            "IC by ticker",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )

    # ── Panel 1: scatter ─────────────────────────────────────────────────────
    for ticker, grp in df.groupby("ticker"):
        fig.add_trace(
            go.Scatter(
                x=grp["signal"], y=grp["return_1d"] * 100,
                mode="markers", name=ticker,
                marker=dict(size=5, opacity=0.6),
            ),
            row=1, col=1,
        )
    fig.update_xaxes(title_text="Sentiment signal (z-score)", row=1, col=1)
    fig.update_yaxes(title_text="Next-day return (%)", row=1, col=1)

    # ── Panel 2: walk-forward IC ─────────────────────────────────────────────
    colors = ["#E24B4A" if v < 0 else "#1D9E75" for v in wf_ic["ic"]]
    fig.add_trace(
        go.Bar(x=wf_ic["date"], y=wf_ic["ic"], marker_color=colors,
               name="Rolling IC", showlegend=False),
        row=1, col=2,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
    fig.update_yaxes(title_text="IC (Spearman)", row=1, col=2)

    # ── Panel 3: cumulative L/S ───────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=ls["date"], y=ls["cumulative_ls"] * 100,
            mode="lines", line=dict(color="#6366f1", width=1.5),
            name="L/S portfolio", showlegend=False,
            fill="tozeroy", fillcolor="rgba(99,102,241,0.1)",
        ),
        row=2, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative return (%)", row=2, col=1)

    # ── Panel 4: IC by ticker ─────────────────────────────────────────────────
    ticker_ics = []
    for ticker, grp in df.groupby("ticker"):
        m = compute_ic(grp)
        ticker_ics.append({"ticker": ticker, "ic": m["ic"]})
    ticker_df = pd.DataFrame(ticker_ics).sort_values("ic", ascending=False)
    bar_colors = ["#E24B4A" if v < 0 else "#1D9E75" for v in ticker_df["ic"]]
    fig.add_trace(
        go.Bar(x=ticker_df["ticker"], y=ticker_df["ic"],
               marker_color=bar_colors, name="Ticker IC", showlegend=False),
        row=2, col=2,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=2)
    fig.update_yaxes(title_text="IC (Spearman)", row=2, col=2)

    fig.update_layout(
        height=650,
        template="plotly_white",
        title="FinBERT Sentiment Alpha — Signal Validation",
        margin=dict(l=50, r=20, t=60, b=40),
    )
    return fig
