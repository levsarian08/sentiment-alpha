"""
FinBERT sentiment scoring for financial headlines.

FinBERT is a BERT model fine-tuned on financial news by ProsusAI.
It outputs three-class probabilities: positive, negative, neutral.

We convert these to a continuous sentiment score in [-1, 1]:
    score = P(positive) - P(negative)

This gives a natural signal: +1 = fully bullish, -1 = fully bearish.

Model: ProsusAI/finbert (auto-downloaded from HuggingFace on first run, ~440MB)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 32
MAX_LENGTH = 128   # headlines are short; 128 is sufficient


class FinBERTScorer:
    """
    Load FinBERT once and score batches of headlines efficiently.

    Attributes
    ----------
    device : torch device (cuda / mps / cpu)
    labels : list of class names in model output order
    """

    def __init__(self):
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Loading FinBERT on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.to(self.device).eval()

        # FinBERT label order: positive=0, negative=1, neutral=2
        self.labels = ["positive", "negative", "neutral"]

    @torch.no_grad()
    def score_batch(self, headlines: list[str]) -> np.ndarray:
        """
        Score a batch of headlines.

        Returns
        -------
        scores : (N,) array of sentiment scores in [-1, 1]
        """
        enc = self.tokenizer(
            headlines,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        # score = P(positive) - P(negative)
        return probs[:, 0] - probs[:, 1]

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a `sentiment` column to a headlines DataFrame.

        Parameters
        ----------
        df : DataFrame with a `headline` column

        Returns
        -------
        df with added columns: sentiment, prob_positive, prob_negative, prob_neutral
        """
        headlines = df["headline"].tolist()
        all_scores = []
        all_probs  = []

        for i in tqdm(range(0, len(headlines), BATCH_SIZE), desc="Scoring headlines"):
            batch = headlines[i : i + BATCH_SIZE]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=MAX_LENGTH, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_scores.extend(probs[:, 0] - probs[:, 1])
            all_probs.extend(probs.tolist())

        result = df.copy()
        result["sentiment"]     = all_scores
        result["prob_positive"] = [p[0] for p in all_probs]
        result["prob_negative"] = [p[1] for p in all_probs]
        result["prob_neutral"]  = [p[2] for p in all_probs]
        return result


if __name__ == "__main__":
    scorer = FinBERTScorer()
    test = [
        "Apple crushes earnings estimates, stock surges after hours",
        "Tesla recalls 200,000 vehicles over safety concerns",
        "Microsoft announces quarterly dividend",
    ]
    scores = scorer.score_batch(test)
    for h, s in zip(test, scores):
        print(f"  {s:+.3f}  {h}")
