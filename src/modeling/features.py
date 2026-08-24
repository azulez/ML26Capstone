"""Aggregates the per-turn sentiment cache into per-transcript tabular features.

Used for the tabular AutoML baseline (Section A of the training notebook) as
a comparison point against the deep attention-pooling model (Section B),
which instead consumes the per-turn embeddings directly (see `dataset.py`).
"""

from __future__ import annotations

import pandas as pd

_SENTIMENT_COLS_PREFIX = "sentiment_"


def aggregate_tabular_features(turn_metadata: pd.DataFrame) -> pd.DataFrame:
    """One row per `row_id`: per-role mean/std sentiment scores + turn counts.

    `turn_metadata` is the `turn_metadata.parquet` written by
    `cli/precompute_turn_features.py` (columns: row_id, turn_idx, speaker,
    title, role, section, n_chars, sentiment_<label>...).
    """
    sentiment_cols = [c for c in turn_metadata.columns if c.startswith(_SENTIMENT_COLS_PREFIX)]

    grouped = turn_metadata.groupby(["row_id", "role"])
    agg = grouped[sentiment_cols].agg(["mean", "std"])
    agg.columns = [f"{role_col}_{stat}" for role_col, stat in agg.columns]
    turn_counts = grouped.size().rename("n_turns")
    per_role = agg.join(turn_counts)

    per_role = per_role.unstack("role")
    per_role.columns = [f"{role}__{col}" for col, role in per_role.columns]
    per_role = per_role.fillna(0.0)

    total_turns = turn_metadata.groupby("row_id").size().rename("total_turns")
    features = per_role.join(total_turns)
    features.index.name = "row_id"
    return features.reset_index()
