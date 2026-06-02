from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from src.schemas.ticket import ClassificationRequest, ClassificationResponse
from src.services.classifier import TicketClassifierService

router = APIRouter(prefix="/api/v1", tags=["classification"])


def get_classifier_service(request: Request) -> TicketClassifierService:
    """Dependency injection provider for the TicketClassifierService.

    Args:
        request: The incoming FastAPI request containing app state.

    Returns:
        The active TicketClassifierService instance.
    """
    return request.app.state.classifier_service


@router.post(
    "/classify",
    response_model=ClassificationResponse,
    summary="Classify a support ticket",
    description="Takes ticket text and predicts its Servicio, Categoria, and Subcategoria.",
)
async def classify_ticket(
    payload: ClassificationRequest,
    service: TicketClassifierService = Depends(get_classifier_service),
) -> ClassificationResponse:
    """Classify an incoming IT support ticket text.

    Args:
        payload: Pydantic request model.
        service: Injected TicketClassifierService instance.

    Returns:
        ClassificationResponse containing predictions.
    """
    predictions = await service.predict(payload.ticket_text)
    return ClassificationResponse(
        ticket_text=payload.ticket_text,
        servicio=predictions["Servicio"],
        categoria=predictions["Categoria"],
        subcategoria=predictions["Subcategoria"],
    )
