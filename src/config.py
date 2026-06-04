from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple

import torch


@dataclass
class Config:
    """Store hyperparameters in one place to keep runs reproducible."""

    model_name: str = "distilbert-base-multilingual-cased"
    data_path: str = "src/data/tickets_soporte_1000 (1).csv"
    output_model_dir: str = "models/distilbert-finetuned/"
    output_tokenizer_dir: str = "models/tokenizer/"
    metrics_path: str = "metrics.json"
    loss_curves_path: str = "loss_curves.png"

    learning_rate: float = 2e-5
    batch_size_gpu: int = 16
    batch_size_cpu: int = 4
    num_epochs: int = 4
    max_length: int = 512
    train_split: float = 0.8

    servicio_classes: int = 5
    categoria_classes: int = 15
    subcategoria_classes: int = 105

    text_column: str = "ticket_text"
    label_columns: Tuple[str, str, str] = ("Servicio", "Categoria", "Subcategoria")

    loss_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "Servicio": 0.2,
            "Categoria": 0.3,
            "Subcategoria": 0.5,
        }
    )
    seed: int = 42

    def resolve_batch_size(self) -> int:
        """Return the batch size chosen for the active device to avoid OOM."""
        return self.batch_size_gpu if torch.cuda.is_available() else self.batch_size_cpu

    def resolve_device(self) -> torch.device:
        """Select the training device so the rest of the code stays simple."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to a dict so it can be logged consistently."""
        return asdict(self)
