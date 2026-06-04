from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from loguru import logger
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.config import Config
from src.model import DistilBertMultiHead
from src.utils import build_label_encoders, load_dataset, tokenize_texts

TASK_KEYS = ("Servicio", "Categoria", "Subcategoria")


class TicketClassifierService:
    """Service class that encapsulates loading the model and running predictions."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize the service with config parameters."""
        self.config = config or Config()
        self.device = self.config.resolve_device()
        self.model: DistilBertMultiHead | None = None
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.decoders: Dict[str, Dict[int, str]] = {}

    def load_model(self) -> None:
        """Load tokenizer, model weights, and decoders from files.

        Raises:
            ModelNotTrainedError: If model files or datasets are missing.
            ModelLoadError: If loading from disk fails for any reason.
        """
        logger.info("Initializing classifier model and resources...")
        state_path = Path(self.config.output_model_dir) / "pytorch_model.bin"

        if not state_path.exists():
            error_msg = f"Model weights not found at {state_path}. Run training first."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            # Load dataset to reconstruct decoders
            logger.info("Reconstructing decoders from dataset...")
            df = load_dataset(self.config.data_path)
            encoders = build_label_encoders(df, self.config.label_columns)
            self.decoders = {
                key: {idx: label for label, idx in mapping.items()}
                for key, mapping in encoders.items()
            }

            # Load tokenizer
            logger.info(f"Loading tokenizer from {self.config.output_tokenizer_dir}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.output_tokenizer_dir)

            # Initialize model architecture and load weights
            logger.info(f"Loading model weights onto device: {self.device.type}...")
            self.model = DistilBertMultiHead(
                self.config.model_name,
                self.config.servicio_classes,
                self.config.categoria_classes,
                self.config.subcategoria_classes,
            )
            state = torch.load(state_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.to(self.device)
            self.model.eval()

            logger.info("Model and tokenizer loaded successfully.")
        except Exception as err:
            logger.exception("Failed to load model resources.")
            raise RuntimeError(f"Unexpected error loading model: {err}") from err

    async def predict(self, text: str) -> Dict[str, str]:
        """Classify a given text into service, category, and subcategory.

        Args:
            text: The ticket description text.

        Returns:
            Dict containing predicted labels for Servicio, Categoria, and Subcategoria.

        Raises:
            PredictionError: If tokenization or inference fails.
        """
        if self.model is None or self.tokenizer is None or not self.decoders:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        try:
            # Run tokenization and move inputs to the same device as the model
            tokens = tokenize_texts([text], self.tokenizer, self.config.max_length)
            inputs = {key: value.to(self.device) for key, value in tokens.items()}

            with torch.no_grad():
                logits = self.model(inputs["input_ids"], inputs["attention_mask"])

            predictions: Dict[str, str] = {}
            for task in TASK_KEYS:
                class_idx = int(logits[task].argmax(dim=1).item())
                predictions[task] = self.decoders[task][class_idx]

            logger.debug(f"Input: {text[:50]}... -> Output: {predictions}")
            return predictions
        except Exception as err:
            logger.exception("Error running prediction model.")
            raise RuntimeError(f"Prediction failed: {err}") from err
