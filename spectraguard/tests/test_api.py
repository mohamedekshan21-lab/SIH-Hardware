"""Integration tests for FastAPI REST API endpoints using TestClient."""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_api_status():
    response = client.get("/api/inspection/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "total_scanned" in data


def test_list_products():
    response = client.get("/api/products?per_page=10")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert len(data["items"]) <= 10


def test_list_lots():
    response = client.get("/api/lots")
    assert response.status_code == 200
    lots = response.json()
    assert isinstance(lots, list)
    assert len(lots) > 0


def test_pdf_report_download():
    # Fetch first lot ID from lots list
    lots_resp = client.get("/api/lots")
    lots = lots_resp.json()
    lot_id = lots[0]["lot_id"]

    pdf_resp = client.get(f"/api/lots/{lot_id}/pdf")
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"] == "application/pdf"
    assert len(pdf_resp.content) > 100


def test_analytics_trend():
    resp = client.get("/api/analytics/trend")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)


def test_audit_log_api():
    resp = client.get("/api/audit-log")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) > 0
