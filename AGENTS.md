# Project: Support Ticket Classifier (DistilBERT fine-tuning)

## Data + outputs
- Dataset lives at `data/tickets_soporte_1000.csv` with columns: `ticket_id`, `ticket_text`, `Servicio`, `Categoria`, `Subcategoria`.
- Training outputs expected: `models/distilbert-finetuned/`, `models/tokenizer/`, `metrics.json`, `training_log.txt`, `loss_curves.png`.

## Model architecture
- Multi-task DistilBERT with 3 classifier heads: Servicio (5 classes, loss weight 0.2), Categoria (15 classes, 0.3), Subcategoria (105 classes, 0.5).

## Hyperparameters (confirmed)
- `learning_rate=2e-5`, `batch_size=16` (GPU) or `4` (CPU), `num_epochs=4`, `max_length=512`, optimizer `AdamW`, split `80/20`.

## Code conventions (non-negotiable)
- Type hints on all functions.
- Docstrings explain what + why.
- Keep functions <= 20 lines and single-responsibility.
- Centralize config in `config.py` (no hardcoding elsewhere).

## Expected src layout
- `src/config.py`, `src/model.py`, `src/utils.py`, `src/train.py`.
