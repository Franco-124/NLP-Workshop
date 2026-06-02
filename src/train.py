from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.config import Config
from src.model import DistilBertMultiHead
from src.utils import (
    build_label_encoders,
    compute_mean,
    compute_metrics,
    create_dataloader,
    encode_labels,
    load_dataset,
    logits_to_predictions,
    split_dataset,
    tokenize_texts,
    to_tensor_labels,
)

INPUT_KEYS = ("input_ids", "attention_mask")
TASK_KEYS = ("Servicio", "Categoria", "Subcategoria")


def log(message: str) -> None:
    """Print a message so progress is visible in the console."""
    print(message, flush=True)


def load_env(path: Path) -> None:
    """Load .env values so runtime settings apply consistently."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"").strip("'"))


def set_seed(seed: int) -> None:
    """Set random seeds to make results reproducible."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    """Create AdamW so fine-tuning matches transformer defaults."""
    return torch.optim.AdamW(model.parameters(), lr=lr)


def compute_weighted_loss(
    logits: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    weights: Dict[str, float],
) -> torch.Tensor:
    """Compute weighted task losses so priorities stay aligned."""
    loss_fn = nn.CrossEntropyLoss()
    total = 0.0
    for task, weight in weights.items():
        total += weight * loss_fn(logits[task], labels[task])
    return total


def move_batch_to_device(
    batch: Dict[str, torch.Tensor], device: torch.device
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Move inputs and labels to device so tensors stay colocated."""
    inputs = {key: batch[key].to(device) for key in INPUT_KEYS}
    labels = {
        key: value.to(device) for key, value in batch.items() if key not in INPUT_KEYS
    }
    return inputs, labels


def train_step(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    weights: Dict[str, float],
) -> float:
    """Run one update step so gradients are applied per batch."""
    logits = model(inputs["input_ids"], inputs["attention_mask"])
    loss = compute_weighted_loss(logits, labels, weights)
    loss.backward()
    optimizer.step()
    return loss.item()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    weights: Dict[str, float],
) -> float:
    """Train for one epoch so loss trends can be tracked."""
    model.train()
    losses: List[float] = []
    for batch in dataloader:
        optimizer.zero_grad()
        inputs, labels = move_batch_to_device(batch, device)
        losses.append(train_step(model, inputs, labels, optimizer, weights))
    return compute_mean(losses)


def log_epoch(epoch: int, num_epochs: int, loss: float) -> None:
    """Report epoch loss so training progress is visible."""
    log(f"Epoch {epoch}/{num_epochs} - loss: {loss:.4f}")


def init_eval_buffers() -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """Initialize buffers so evaluation stays consistent."""
    y_true = {task: [] for task in TASK_KEYS}
    y_pred = {task: [] for task in TASK_KEYS}
    return y_true, y_pred


def update_eval_buffers(
    y_true: Dict[str, List[int]],
    y_pred: Dict[str, List[int]],
    labels: Dict[str, torch.Tensor],
    logits: Dict[str, torch.Tensor],
) -> None:
    """Append batch predictions so metrics can be computed later."""
    for task in TASK_KEYS:
        y_true[task].extend(labels[task].cpu().tolist())
        y_pred[task].extend(logits_to_predictions(logits[task]))


def eval_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """Evaluate a full epoch so metrics reflect all samples."""
    model.eval()
    y_true, y_pred = init_eval_buffers()
    log("Evaluating on validation set...")
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = move_batch_to_device(batch, device)
            logits = model(inputs["input_ids"], inputs["attention_mask"])
            update_eval_buffers(y_true, y_pred, labels, logits)
    return y_true, y_pred


def prepare_labels(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    label_columns: Tuple[str, str, str],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Encode labels once so train/valid share consistent ids."""
    encoders = build_label_encoders(train_df, label_columns)
    train_labels = encode_labels(train_df, encoders)
    valid_labels = encode_labels(valid_df, encoders)
    return to_tensor_labels(train_labels), to_tensor_labels(valid_labels)


def prepare_inputs(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    text_column: str,
    max_length: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Tokenize both splits so they share the same settings."""
    train_inputs = tokenize_texts(train_df[text_column].tolist(), tokenizer, max_length)
    valid_inputs = tokenize_texts(valid_df[text_column].tolist(), tokenizer, max_length)
    return train_inputs, valid_inputs


def load_and_split_data(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and split data so the pipeline stays deterministic."""
    df = load_dataset(config.data_path)
    log(f"Loaded dataset: {len(df)} rows from {config.data_path}")
    train_df, valid_df = split_dataset(df, config.train_split, config.seed)
    log(f"Train rows: {len(train_df)} | Valid rows: {len(valid_df)}")
    return train_df, valid_df


def build_dataloaders(
    train_inputs: Dict[str, torch.Tensor],
    valid_inputs: Dict[str, torch.Tensor],
    train_labels: Dict[str, torch.Tensor],
    valid_labels: Dict[str, torch.Tensor],
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """Create dataloaders so training and validation are consistent."""
    train_loader = create_dataloader(train_inputs, train_labels, batch_size, True)
    valid_loader = create_dataloader(valid_inputs, valid_labels, batch_size, False)
    return train_loader, valid_loader


def prepare_data(
    config: Config,
) -> Tuple[DataLoader, DataLoader, PreTrainedTokenizerBase]:
    """Prepare tokenized loaders so the rest of the pipeline is clean."""
    train_df, valid_df = load_and_split_data(config)
    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    log("Tokenizing texts...")
    train_inputs, valid_inputs = prepare_inputs(
        train_df, valid_df, tokenizer, config.text_column, config.max_length
    )
    train_labels, valid_labels = prepare_labels(train_df, valid_df, config.label_columns)
    batch_size = config.resolve_batch_size()
    log(f"Batch size: {batch_size}")
    loaders = build_dataloaders(train_inputs, valid_inputs, train_labels, valid_labels, batch_size)
    return loaders[0], loaders[1], tokenizer


def build_model(config: Config) -> DistilBertMultiHead:
    """Build the multi-task model so heads match class counts."""
    return DistilBertMultiHead(
        config.model_name,
        config.servicio_classes,
        config.categoria_classes,
        config.subcategoria_classes,
    )


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    weights: Dict[str, float],
    num_epochs: int,
) -> List[float]:
    """Run epochs so loss history is available for reporting."""
    losses: List[float] = []
    for epoch in range(1, num_epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device, weights)
        losses.append(loss)
        log_epoch(epoch, num_epochs, loss)
    return losses


def evaluate_model(
    model: nn.Module, valid_loader: DataLoader, device: torch.device
) -> Dict[str, Dict[str, float]]:
    """Evaluate model so per-task metrics can be saved."""
    y_true, y_pred = eval_epoch(model, valid_loader, device)
    return compute_metrics(y_true, y_pred)


def log_metrics(metrics: Dict[str, Dict[str, float]]) -> None:
    """Print metrics so evaluation results are visible."""
    log("Validation metrics:")
    for task, values in metrics.items():
        log(
            f"{task} | f1={values['f1']:.4f} "
            f"precision={values['precision']:.4f} recall={values['recall']:.4f}"
        )


def save_metrics(path: str, metrics: Dict[str, Dict[str, float]]) -> None:
    """Save metrics to JSON so evaluation is traceable."""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def save_loss_history(path: str, loss_history: List[float]) -> None:
    """Save loss history so training can be inspected later."""
    with open(path, "w", encoding="utf-8") as file:
        json.dump({"loss_history": loss_history}, file, indent=2)


def save_loss_plot(path: str, loss_history: List[float]) -> None:
    """Plot training loss so trends are easy to visualize."""
    plt.figure()
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(path)
    plt.close()


def save_tokenizer(tokenizer: PreTrainedTokenizerBase, output_dir: str) -> None:
    """Persist tokenizer so inference uses the same vocabulary."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)


def save_artifacts(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    config: Config,
    metrics: Dict[str, Dict[str, float]],
    loss_history: List[float],
) -> None:
    """Save outputs so training results are reproducible."""
    log("Saving artifacts...")
    model.save_pretrained(config.output_model_dir)
    save_tokenizer(tokenizer, config.output_tokenizer_dir)
    save_metrics(config.metrics_path, metrics)
    save_loss_history(config.training_log_path, loss_history)
    save_loss_plot(config.loss_curves_path, loss_history)


def main() -> None:
    """Run the full fine-tuning pipeline to produce required artifacts."""
    load_env(Path(__file__).resolve().parents[1] / ".env")
    config = Config()
    set_seed(config.seed)
    device = config.resolve_device()
    log("Starting training...")
    log(f"Model: {config.model_name}")
    log(f"Device: {device.type}")
    train_loader, valid_loader, tokenizer = prepare_data(config)
    model = build_model(config).to(device)
    log("Building optimizer...")
    optimizer = build_optimizer(model, config.learning_rate)
    log("Training epochs...")
    loss_history = run_training(
        model, train_loader, optimizer, device, config.loss_weights, config.num_epochs
    )
    log("Computing metrics...")
    metrics = evaluate_model(model, valid_loader, device)
    log_metrics(metrics)
    save_artifacts(model, tokenizer, config, metrics, loss_history)
    log("Done.")


if __name__ == "__main__":
    main()
