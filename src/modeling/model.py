"""The speaker-turn attention-pooling model (see docs/model_architecture.md).

Consumes precomputed, frozen FinBERT turn embeddings (see `encoding.py`) —
"fine-tuning" here means training the attention-pooling head end-to-end
against `excess_return_label`, not unfreezing the FinBERT encoder itself.
Re-running FinBERT during training would reintroduce the expensive forward
pass this project's caching step (`cli/precompute_turn_features.py`) exists
to avoid, and the head is the only part of the model with anywhere to learn
turn/speaker-role importance anyway (see the supervision-caveat section of
the architecture doc: there is no per-turn label to fine-tune the encoder
against).
"""

from __future__ import annotations

import torch
from torch import nn

from .dataset import NUM_ROLE_IDS, PAD_ROLE_ID


class AttentionPoolingClassifier(nn.Module):
    """Role-conditioned additive attention over turn embeddings -> document
    embedding -> binary classification head.

    Attention weights are returned alongside logits: they're the "which
    speaker/turn drove this prediction" signal from the design discussion.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        role_dim: int = 16,
        attention_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.role_embedding = nn.Embedding(NUM_ROLE_IDS, role_dim, padding_idx=PAD_ROLE_ID)
        turn_repr_dim = embedding_dim + role_dim

        self.turn_proj = nn.Sequential(
            nn.Linear(turn_repr_dim, attention_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.attention_score = nn.Linear(attention_dim, 1, bias=False)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(attention_dim, 1),
        )

    def forward(
        self, embeddings: torch.Tensor, role_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`embeddings`: [B, T, embedding_dim], `role_ids`/`attention_mask`: [B, T].
        Returns (logits [B], attention_weights [B, T])."""
        role_embeds = self.role_embedding(role_ids)  # [B, T, role_dim]
        turn_repr = torch.cat([embeddings, role_embeds], dim=-1)  # [B, T, turn_repr_dim]
        projected = self.turn_proj(turn_repr)  # [B, T, attention_dim]

        scores = self.attention_score(projected).squeeze(-1)  # [B, T]
        scores = scores.masked_fill(~attention_mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)  # [B, T]

        pooled = torch.einsum("bt,btd->bd", attn_weights, projected)  # [B, attention_dim]
        logits = self.classifier(pooled).squeeze(-1)  # [B]
        return logits, attn_weights
