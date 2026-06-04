from __future__ import annotations

from typing import Dict, Generator

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.services.classifier import TicketClassifierService


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Provide a TestClient with service class mocked to prevent disk/network IO."""
    # Store original methods to restore them later
    original_load = TicketClassifierService.load_model
    original_predict = TicketClassifierService.predict

    # Mock load_model to do nothing
    TicketClassifierService.load_model = lambda self: None

    # Mock predict method to return static values
    async def mock_predict(self, text: str) -> Dict[str, str]:
        return {
            "Servicio": "Mocked Service",
            "Categoria": "Mocked Category",
            "Subcategoria": "Mocked Subcategory",
        }

    TicketClassifierService.predict = mock_predict

    # Create the client (this triggers the standard startup lifespan event)
    with TestClient(app) as test_client:
        yield test_client

    # Restore original methods after the test
    TicketClassifierService.load_model = original_load
    TicketClassifierService.predict = original_predict


def test_health_check(client: TestClient) -> None:
    """Test that the health check endpoint returns 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_classify_success(client: TestClient) -> None:
    """Test that classifying a valid ticket text succeeds."""
    payload = {"ticket_text": "No puedo iniciar sesión en mi cuenta."}
    response = client.post("/classify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["ticket_text"] == payload["ticket_text"]
    assert data["servicio"] == "Mocked Service"
    assert data["categoria"] == "Mocked Category"
    assert data["subcategoria"] == "Mocked Subcategory"


def test_classify_validation_error(client: TestClient) -> None:
    """Test that classifying a text that is too short returns a validation error."""
    payload = {"ticket_text": "Hi"}  # shorter than 3 chars limit
    response = client.post("/classify", json=payload)
    assert response.status_code == 422  # Unprocessable Entity
