"""Train/eval loop, time-based data splitting, and verification-plot helpers
for the attention-pooling model (see model.py) and shared by both notebook
sections for a consistent evaluation story.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader


def time_based_split(
    labels_df: pd.DataFrame,
    date_col: str = "call_date",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits chronologically (not randomly) to avoid look-ahead leakage: a
    model trained on later calls "seeing" earlier calls' market reactions
    would understate real-world deployment error, since in production you
    only ever have data up to the call being scored.
    """
    ordered = labels_df.sort_values(date_col)
    n = len(ordered)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train_df = ordered.iloc[:train_end]
    val_df = ordered.iloc[train_end:val_end]
    test_df = ordered.iloc[val_end:]
    return train_df, val_df, test_df


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    optimizer: torch.optim.Optimizer | None,
    loss_fn: nn.Module,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            embeddings = batch["embeddings"].to(device)
            role_ids = batch["role_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits, _ = model(embeddings, role_ids, attention_mask)
            loss = loss_fn(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = labels.shape[0]
            total_loss += loss.item() * batch_size
            total_correct += ((logits > 0).float() == labels).sum().item()
            total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    patience: int | None = 5,
) -> TrainingHistory:
    """Trains with optional early stopping on val loss. `patience` is itself
    an over/underfitting guard — this is where the notebook's `MAX_EPOCHS`
    depth-limiting constant applies as an upper bound.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    history = TrainingHistory()

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for _ in range(max_epochs):
        train_loss, train_acc = _run_epoch(model, train_loader, device, optimizer, loss_fn)
        val_loss, val_acc = _run_epoch(model, val_loader, device, None, loss_fn)

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.train_acc.append(train_acc)
        history.val_acc.append(val_acc)

        if patience is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break

    return history


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str = "cpu") -> tuple[list[int], list[int]]:
    """Returns (y_true, y_pred) as 0/1 lists over the full loader."""
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        embeddings = batch["embeddings"].to(device)
        role_ids = batch["role_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        logits, _ = model(embeddings, role_ids, attention_mask)
        y_pred.extend((logits.cpu() > 0).int().tolist())
        y_true.extend(labels.int().tolist())
    return y_true, y_pred


def plot_training_curves(history: TrainingHistory, title: str = "") -> plt.Figure:
    """Side-by-side train-vs-val loss and accuracy curves — the direct visual
    check for over/underfitting: a widening gap between train and val loss is
    overfitting, both curves plateauing high is underfitting.
    """
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history.train_loss) + 1)

    loss_ax.plot(epochs, history.train_loss, marker="o", label="train")
    loss_ax.plot(epochs, history.val_loss, marker="o", label="val")
    loss_ax.set_xlabel("epoch")
    loss_ax.set_ylabel("loss")
    loss_ax.set_title("Loss")
    loss_ax.legend()

    acc_ax.plot(epochs, history.train_acc, marker="o", label="train")
    acc_ax.plot(epochs, history.val_acc, marker="o", label="val")
    acc_ax.set_xlabel("epoch")
    acc_ax.set_ylabel("accuracy")
    acc_ax.set_title("Accuracy")
    acc_ax.legend()

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(y_true: list[int], y_pred: list[int], title: str = "") -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["pred: underperform", "pred: outperform"],
        yticklabels=["true: underperform", "true: outperform"],
        ax=ax,
        cbar=False,
    )
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig
