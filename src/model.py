from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch import nn
from transformers import AutoModel


class DistilBertMultiHead(nn.Module):
    """Share a DistilBERT backbone to learn all tasks jointly."""

    def __init__(
        self,
        model_name: str,
        num_servicio: int,
        num_categoria: int,
        num_subcategoria: int,
    ) -> None:
        """Create backbone and heads so tasks can share representations."""
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(self.backbone.config.dropout)
        self.head_servicio = nn.Linear(hidden_size, num_servicio)
        self.head_categoria = nn.Linear(hidden_size, num_categoria)
        self.head_subcategoria = nn.Linear(hidden_size, num_subcategoria)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Return logits per task so each loss can be computed separately."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        return {
            "Servicio": self.head_servicio(pooled),
            "Categoria": self.head_categoria(pooled),
            "Subcategoria": self.head_subcategoria(pooled),
        }

    def save_pretrained(self, output_dir: str) -> None:
        """Persist weights and config so the fine-tuned model can be reused."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(output_dir) / "pytorch_model.bin")
        self.backbone.config.save_pretrained(output_dir)
