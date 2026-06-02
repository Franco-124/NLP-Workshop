from __future__ import annotations


class ClassifierException(Exception):
    """Base exception for all classifier-related errors."""

    def __init__(self, message: str) -> None:
        """Initialize custom exception with a message."""
        super().__init__(message)
        self.message = message


class ModelLoadError(ClassifierException):
    """Raised when the model or tokenizer fails to load from disk."""


class ModelNotTrainedError(ClassifierException):
    """Raised when the trained model artifacts are missing."""


class PredictionError(ClassifierException):
    """Raised when inference fails due to input format or runtime errors."""
