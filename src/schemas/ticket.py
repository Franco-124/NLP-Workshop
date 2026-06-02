from __future__ import annotations

from pydantic import BaseModel, Field


class ClassificationRequest(BaseModel):
    """Schema representing a ticket classification request."""

    ticket_text: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        examples=["No puedo acceder a mi máquina virtual desde la VPN."],
        description="Text content of the IT support ticket to be classified.",
    )


class ClassificationResponse(BaseModel):
    """Schema representing the classification results."""

    ticket_text: str = Field(
        ...,
        description="The original ticket text that was analyzed.",
    )
    servicio: str = Field(
        ...,
        description="Predicted macro service category.",
    )
    categoria: str = Field(
        ...,
        description="Predicted medium-level category.",
    )
    subcategoria: str = Field(
        ...,
        description="Predicted micro-level subcategory.",
    )
