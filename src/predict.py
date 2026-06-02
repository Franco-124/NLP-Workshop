from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.config import Config
from src.model import DistilBertMultiHead
from src.utils import build_label_encoders, load_dataset, tokenize_texts

TASK_KEYS = ("Servicio", "Categoria", "Subcategoria")


def log(message: str) -> None:
    """Print a message so prediction flow is visible."""
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


def parse_args() -> argparse.Namespace:
    """Parse CLI args so interactive or one-shot modes are supported."""
    parser = argparse.ArgumentParser(description="Ticket classifier CLI")
    parser.add_argument("--text", type=str, help="Ticket text to classify")
    return parser.parse_args()


def ensure_file(path: Path) -> None:
    """Raise a clear error if a required file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


def build_decoders(
    df: pd.DataFrame, label_columns: Tuple[str, str, str]
) -> Dict[str, Dict[int, str]]:
    """Build decoders so predictions map back to labels."""
    encoders = build_label_encoders(df, label_columns)
    return {key: {idx: label for label, idx in mapping.items()} for key, mapping in encoders.items()}


def build_model(config: Config, device: torch.device) -> DistilBertMultiHead:
    """Initialize model so heads match the configured class counts."""
    model = DistilBertMultiHead(
        config.model_name,
        config.servicio_classes,
        config.categoria_classes,
        config.subcategoria_classes,
    )
    return model.to(device)


def load_model_state(
    model: DistilBertMultiHead, state_path: Path, device: torch.device
) -> None:
    """Load model weights so predictions use trained parameters."""
    state = torch.load(state_path, map_location=device)
    model.load_state_dict(state)


def prepare_model(config: Config, device: torch.device) -> DistilBertMultiHead:
    """Load model state and set eval mode for inference."""
    model = build_model(config, device)
    state_path = Path(config.output_model_dir) / "pytorch_model.bin"
    ensure_file(state_path)
    load_model_state(model, state_path, device)
    model.eval()
    return model


def prepare_tokenizer(config: Config) -> PreTrainedTokenizerBase:
    """Load tokenizer so inputs match training vocabulary."""
    return AutoTokenizer.from_pretrained(config.output_tokenizer_dir)


def predict_text(
    model: DistilBertMultiHead,
    tokenizer: PreTrainedTokenizerBase,
    decoders: Dict[str, Dict[int, str]],
    text: str,
    max_length: int,
    device: torch.device,
) -> Dict[str, str]:
    """Predict labels for one text so outputs are human-readable."""
    tokens = tokenize_texts([text], tokenizer, max_length)
    inputs = {key: value.to(device) for key, value in tokens.items()}
    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
    return {
        task: decoders[task][int(logits[task].argmax(dim=1).item())]
        for task in TASK_KEYS
    }


def format_output(preds: Dict[str, str]) -> str:
    """Format predictions so CLI output is compact."""
    return (
        f"Servicio: {preds['Servicio']} | "
        f"Categoria: {preds['Categoria']} | "
        f"Subcategoria: {preds['Subcategoria']}"
    )


def interactive_loop(
    model: DistilBertMultiHead,
    tokenizer: PreTrainedTokenizerBase,
    decoders: Dict[str, Dict[int, str]],
    config: Config,
    device: torch.device,
) -> None:
    """Read user input so multiple tickets can be classified quickly."""
    log("Interactive mode. Submit empty input to exit.")
    while True:
        text = input("Ticket text: ").strip()
        if not text:
            log("Bye.")
            return
        preds = predict_text(model, tokenizer, decoders, text, config.max_length, device)
        log(format_output(preds))


def main() -> None:
    """Run the CLI to classify tickets interactively or once."""
    load_env(Path(__file__).resolve().parents[1] / ".env")
    args = parse_args()
    config = Config()
    device = config.resolve_device()
    df = load_dataset(config.data_path)
    decoders = build_decoders(df, config.label_columns)
    tokenizer = prepare_tokenizer(config)
    model = prepare_model(config, device)
    if args.text:
        preds = predict_text(model, tokenizer, decoders, args.text, config.max_length, device)
        log(format_output(preds))
        return
    interactive_loop(model, tokenizer, decoders, config, device)


if __name__ == "__main__":
    main()
