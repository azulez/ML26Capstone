"""Batched FinBERT encoding of speaker turns into embeddings + sentiment scores.

One forward pass per turn produces both outputs used downstream: a pooled
embedding (input to the deep attention-pooling model, see `model.py`) and
sentiment class probabilities (input to the tabular AutoML feature
aggregation, see `features.py`). Precomputing both from a single pass avoids
running FinBERT twice over the same ~230k turns in the dataset.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("HF_HUB_OFFLINE", "0")

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class EncodedTurns:
    embeddings: np.ndarray  # [N, hidden_size], float32
    sentiment_probs: np.ndarray  # [N, num_labels], float32
    sentiment_labels: list[str]  # length num_labels, e.g. ["positive", "negative", "neutral"]


class TurnEncoder:
    """Wraps ProsusAI/finbert (consistent with test/finbert_sentiment.py) for batched inference."""

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 256,
    ):
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._batch_size = batch_size
        self._max_length = max_length
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name, output_hidden_states=True
        )
        self._model.eval().to(self._device)
        num_labels = self._model.config.num_labels
        self.sentiment_labels = [self._model.config.id2label[i] for i in range(num_labels)]

    @torch.no_grad()
    def encode(self, texts: list[str], show_progress: bool = False) -> EncodedTurns:
        if not texts:
            hidden_size = self._model.config.hidden_size
            return EncodedTurns(
                embeddings=np.empty((0, hidden_size), dtype=np.float32),
                sentiment_probs=np.empty((0, len(self.sentiment_labels)), dtype=np.float32),
                sentiment_labels=self.sentiment_labels,
            )

        all_embeddings = []
        all_probs = []
        batch_starts = range(0, len(texts), self._batch_size)
        if show_progress:
            batch_starts = tqdm(list(batch_starts), desc="Encoding turns", unit="batch")
        for start in batch_starts:
            batch = texts[start : start + self._batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            ).to(self._device)

            outputs = self._model(**encoded)
            probs = torch.softmax(outputs.logits, dim=-1)

            last_hidden = outputs.hidden_states[-1]  # [B, T, H]
            mask = encoded["attention_mask"].unsqueeze(-1).float()  # [B, T, 1]
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-6)
            pooled = summed / counts  # [B, H]

            all_embeddings.append(pooled.cpu().numpy().astype(np.float32))
            all_probs.append(probs.cpu().numpy().astype(np.float32))

        return EncodedTurns(
            embeddings=np.concatenate(all_embeddings, axis=0),
            sentiment_probs=np.concatenate(all_probs, axis=0),
            sentiment_labels=self.sentiment_labels,
        )
