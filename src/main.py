from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.schemas.ticket import ClassificationRequest, ClassificationResponse
from src.services.classifier import TicketClassifierService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the ML model into memory on startup and clean up on shutdown."""
    logger.info("FastAPI starting: Loading classifier model...")
    service = TicketClassifierService()
    service.load_model()
    app.state.classifier_service = service
    yield
    if hasattr(app.state, "classifier_service"):
        del app.state.classifier_service


app = FastAPI(
    title="IT Support Ticket Classifier API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post(
    "/classify",
    response_model=ClassificationResponse,
    summary="Classify ticket text",
)
async def classify_ticket(
    payload: ClassificationRequest,
    request: Request,
) -> ClassificationResponse:
    """Clasifica un ticket en Servicio, Categoría y Subcategoría."""
    service: TicketClassifierService = request.app.state.classifier_service
    predictions = await service.predict(payload.ticket_text)
    return ClassificationResponse(
        ticket_text=payload.ticket_text,
        servicio=predictions["Servicio"],
        categoria=predictions["Categoria"],
        subcategoria=predictions["Subcategoria"],
    )


@app.get("/health", tags=["monitoring"])
async def health_check() -> dict[str, str]:
    """Health check endpoint to verify server is running."""
    return {"status": "ok"}
