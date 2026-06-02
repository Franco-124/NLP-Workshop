from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase


def load_dataset(path: str) -> pd.DataFrame:
    """Load the CSV so downstream steps share a single IO entrypoint."""
    return pd.read_csv(path)


def split_dataset(
    df: pd.DataFrame, train_ratio: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split with a fixed seed so train/valid is reproducible."""
    train_df = df.sample(frac=train_ratio, random_state=seed)
    valid_df = df.drop(train_df.index)
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)


def build_label_encoders(
    df: pd.DataFrame, label_columns: Iterable[str]
) -> Dict[str, Dict[str, int]]:
    """Create label maps per task so ids stay stable across runs."""
    encoders: Dict[str, Dict[str, int]] = {}
    for column in label_columns:
        labels = sorted(df[column].dropna().unique().tolist())
        encoders[column] = {label: idx for idx, label in enumerate(labels)}
    return encoders


def encode_labels(
    df: pd.DataFrame, encoders: Dict[str, Dict[str, int]]
) -> Dict[str, List[int]]:
    """Map string labels to indices so they are loss-ready."""
    encoded: Dict[str, List[int]] = {}
    for column, mapping in encoders.items():
        encoded[column] = [mapping[label] for label in df[column].tolist()]
    return encoded


def tokenize_texts(
    texts: List[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    """Tokenize texts so inputs match DistilBERT expectations."""
    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]}


@dataclass
class TicketDataset(Dataset):
    """Bundle inputs and labels so DataLoader stays simple."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: Dict[str, torch.Tensor]

    def __len__(self) -> int:
        """Return dataset size so DataLoader can iterate correctly."""
        return self.input_ids.size(0)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Return one example so DataLoader can build batches."""
        item = {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
        }
        for key, value in self.labels.items():
            item[key] = value[index]
        return item


def create_dataloader(
    inputs: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create DataLoader so batching logic stays consistent."""
    dataset = TicketDataset(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        labels=labels,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_metrics(
    y_true: Dict[str, List[int]], y_pred: Dict[str, List[int]]
) -> Dict[str, Dict[str, float]]:
    """Compute macro metrics per task to compare tasks fairly."""
    metrics: Dict[str, Dict[str, float]] = {}
    for task, true_values in y_true.items():
        pred_values = y_pred[task]
        metrics[task] = {
            "f1": float(f1_score(true_values, pred_values, average="macro")),
            "precision": float(
                precision_score(true_values, pred_values, average="macro")
            ),
            "recall": float(recall_score(true_values, pred_values, average="macro")),
        }
    return metrics


def to_tensor_labels(labels: Dict[str, List[int]]) -> Dict[str, torch.Tensor]:
    """Convert label lists to tensors so Dataset can index them."""
    return {key: torch.tensor(values, dtype=torch.long) for key, values in labels.items()}


def logits_to_predictions(logits: torch.Tensor) -> List[int]:
    """Convert logits to class indices for metric computation."""
    return logits.argmax(dim=1).cpu().tolist()


def compute_mean(values: List[float]) -> float:
    """Compute mean safely to keep logging concise."""
    return float(np.mean(values)) if values else 0.0
