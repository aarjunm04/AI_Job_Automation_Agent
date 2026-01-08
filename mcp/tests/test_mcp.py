# tests/test_mcp.py
# Minimal test suite for MCP service

import pytest
from fastapi.testclient import TestClient
from mcp_client import app, mcp_service
import os

@pytest.fixture(scope="module", autouse=True)
def setup_env():
    # Enable dev mode to disable auth
    os.environ["MCP_DEV_MODE"] = "1"
    yield

@pytest.fixture(scope="module")
def client():
    return TestClient(app)

def test_create_session(client):
    resp = client.post("/v1/sessions", json={"owner": "pytest"})
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    return data["session_id"]

def test_append_item(client):
    session = client.post("/v1/sessions", json={"owner": "pytest2"}).json()
    sid = session["session_id"]

    item = {
        "role": "tool",
        "content": "Test Content",
        "metadata": {"source": "pytest"}
    }
    resp = client.post(f"/v1/sessions/{sid}/items", json=item)
    assert resp.status_code == 200
    assert resp.json()["content"] == "Test Content"

def test_get_items(client):
    session = client.post("/v1/sessions", json={"owner": "pytest3"}).json()
    sid = session["session_id"]

    client.post(f"/v1/sessions/{sid}/items", json={"role": "tool", "content": "A"})
    client.post(f"/v1/sessions/{sid}/items", json={"role": "tool", "content": "B"})

    resp = client.get(f"/v1/sessions/{sid}/items?last_n=2")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert len(data["items"]) == 2

def test_snapshot(client):
    session = client.post("/v1/sessions", json={"owner": "pytest4"}).json()
    sid = session["session_id"]

    # Add some context
    client.post(f"/v1/sessions/{sid}/items", json={"role": "tool", "content": "hello world"})
    client.post(f"/v1/sessions/{sid}/items", json={"role": "tool", "content": "machine learning jobs"})
    client.post(f"/v1/sessions/{sid}/items", json={"role": "tool", "content": "python ai tooling"})

    # Create snapshot
    resp = client.post(f"/v1/sessions/{sid}/snapshot", json={"strategy": "rolling", "max_sentences": 5})
    assert resp.status_code == 200
    snap = resp.json()
    assert "summary_text" in snap
    assert snap["session_id"] == sid