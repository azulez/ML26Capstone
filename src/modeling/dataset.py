"""PyTorch Dataset over the precomputed per-turn embedding cache, grouped by transcript.

Each item is one transcript's full set of turn embeddings + role tags;
`collate_fn` pads variable turn counts to the batch max so the attention-pooling
model (`model.py`) can consume a batch at once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .turns import Role

ROLE_VOCAB: dict[str, int] = {role.value: i for i, role in enumerate(Role)}
PAD_ROLE_ID = len(ROLE_VOCAB)
NUM_ROLE_IDS = len(ROLE_VOCAB) + 1  # + 1 for the pad id


class TurnFeatureDataset(Dataset):
    """`turn_metadata` + `embeddings` are the two artifacts written by
    `cli/precompute_turn_features.py` (row-aligned). `labels` maps row_id ->
    a 0/1 (or float) target, and only row_ids present in `labels` are kept —
    this is how the notebook applies `SAMPLE_FRACTION` and the train/val/test
    split before constructing the dataset.
    """

    def __init__(self, turn_metadata: pd.DataFrame, embeddings: np.ndarray, labels: pd.Series):
        # `turn_metadata` must be the full, unfiltered cache with its original
        # RangeIndex (positions == index labels), since positions are used to
        # index directly into the row-aligned `embeddings` array. Subsample by
        # restricting `labels` to the row_ids you want, not by slicing
        # `turn_metadata` itself.
        self._labels = labels
        self.row_ids = list(labels.index)

        role_ids = turn_metadata["role"].map(ROLE_VOCAB).to_numpy()
        self._indices_by_row: dict[int, np.ndarray] = {
            row_id: positions.to_numpy()
            for row_id, positions in turn_metadata.groupby("row_id").groups.items()
        }
        self._embeddings = embeddings
        self._role_ids = role_ids

    def __len__(self) -> int:
        return len(self.row_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row_id = self.row_ids[idx]
        turn_positions = self._indices_by_row.get(row_id, np.empty(0, dtype=int))
        embeddings = torch.from_numpy(self._embeddings[turn_positions]).float()
        role_ids = torch.from_numpy(self._role_ids[turn_positions]).long()
        label = torch.tensor(float(self._labels.loc[row_id]), dtype=torch.float32)
        return embeddings, role_ids, label


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    embeddings, role_ids, labels = zip(*batch)
    hidden_size = embeddings[0].shape[-1]
    max_turns = max(e.shape[0] for e in embeddings)
    batch_size = len(embeddings)

    padded_embeddings = torch.zeros(batch_size, max_turns, hidden_size, dtype=torch.float32)
    padded_role_ids = torch.full((batch_size, max_turns), PAD_ROLE_ID, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_turns, dtype=torch.bool)

    for i, (emb, roles) in enumerate(zip(embeddings, role_ids)):
        n = emb.shape[0]
        padded_embeddings[i, :n] = emb
        padded_role_ids[i, :n] = roles
        attention_mask[i, :n] = True

    return {
        "embeddings": padded_embeddings,
        "role_ids": padded_role_ids,
        "attention_mask": attention_mask,
        "labels": torch.stack(labels),
    }
