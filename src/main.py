from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from src.api.routes import router
from src.core.exceptions import ClassifierException, ModelNotTrainedError
from src.services.classifier import TicketClassifierService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage startup and shutdown lifecycles of the application.

    Initializes the model weights and label decoders on start,
    and cleans up references on shutdown.
    """
    logger.info("Starting up FastAPI application...")
    classifier_service = TicketClassifierService()
    try:
        classifier_service.load_model()
        app.state.classifier_service = classifier_service
        logger.info("FastAPI initialization completed.")
        yield
    except ModelNotTrainedError:
        logger.warning(
            "Application started without a trained model. "
            "Please train the model using: uv run python -m src.train"
        )
        app.state.classifier_service = classifier_service
        yield
    except Exception as err:
        logger.critical(f"Failed to start FastAPI server: {err}")
        raise err
    finally:
        logger.info("Shutting down FastAPI application...")
        if hasattr(app.state, "classifier_service"):
            del app.state.classifier_service


app = FastAPI(
    title="IT Support Ticket Classifier API",
    description="Multi-task ticket classification using fine-tuned DistilBERT.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware to allow requests from frontend interfaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Log execution latency and assign correlation ids if needed."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
    logger.info(
        f"Path: {request.url.path} | Status: {response.status_code} | Latency: {process_time:.4f}s"
    )
    return response


@app.exception_handler(ClassifierException)
async def classifier_exception_handler(
    request: Request, exc: ClassifierException
) -> JSONResponse:
    """Global handler for domain-specific classification exceptions."""
    logger.error(f"Classifier error at {request.url.path}: {exc.message}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": exc.message, "code": "CLASSIFIER_ERROR"},
    )


@app.exception_handler(ModelNotTrainedError)
async def model_not_trained_handler(
    request: Request, exc: ModelNotTrainedError
) -> JSONResponse:
    """Specialized handler when endpoints are hit but the model files do not exist."""
    logger.error(f"Missing model files at {request.url.path}: {exc.message}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "detail": "Model is not trained or loaded. Please run the training pipeline.",
            "code": "MODEL_NOT_TRAINED",
        },
    )


# Health check endpoint
@app.get("/health", status_code=status.HTTP_200_OK, tags=["monitoring"])
async def health_check() -> dict[str, str]:
    """Retrieve system health status."""
    return {"status": "ok", "message": "API is online and accepting requests."}


# Include business API routes
app.include_router(router)
