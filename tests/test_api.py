import json

import pytest
import requests

import generator
import main
from conftest import FakeResponse, FakeSession, chat_response
from tests_data import VALID_SCRIPT


@pytest.fixture
def client():
    main.app.config["TESTING"] = True
    return main.app.test_client()


@pytest.fixture
def upstream(monkeypatch):
    """Route the module-level `requests.post` through a FakeSession."""

    def install(outcomes):
        session = FakeSession(outcomes)
        monkeypatch.setattr(generator.requests, "post", session.post)
        return session

    return install


def test_home_reports_service_status(client):
    body = client.get("/").get_json()
    assert body["status"] == "ok"
    assert body["configured"] is True
    assert "POST /generate" in body["endpoints"]


def test_healthz_stays_200_without_a_key(client, monkeypatch):
    monkeypatch.delenv("HF_API_KEY", raising=False)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.get_json()["configured"] is False


def test_generate_returns_structured_script(client, upstream):
    session = upstream([chat_response(json.dumps(VALID_SCRIPT))])
    response = client.post("/generate", json={"prompt": "A house that counts",
                                              "duration_minutes": 12})
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "success"
    assert body["script"]["title"] == VALID_SCRIPT["title"]
    assert body["script"]["format"] == "structured"
    assert body["request"]["duration_minutes"] == 12
    assert len(session.calls) == 1


def test_generate_works_with_no_body(client, upstream):
    upstream([chat_response(json.dumps(VALID_SCRIPT))])
    response = client.post("/generate")
    assert response.status_code == 200


def test_generate_rejects_non_object_body(client):
    response = client.post("/generate", json=["not", "an", "object"])
    assert response.status_code == 400
    assert response.get_json()["error"] == "invalid_request"


def test_generate_rejects_invalid_duration(client):
    response = client.post("/generate", json={"duration_minutes": 999})
    assert response.status_code == 400
    body = response.get_json()
    assert body["error"] == "invalid_request"
    assert "duration_minutes" in body["message"]


def test_generate_returns_503_when_unconfigured(client, monkeypatch):
    monkeypatch.delenv("HF_API_KEY", raising=False)
    response = client.post("/generate", json={"prompt": "anything"})
    assert response.status_code == 503
    assert response.get_json()["error"] == "not_configured"


def test_generate_returns_502_on_upstream_failure(client, upstream, no_sleep, monkeypatch):
    monkeypatch.setenv("HF_MAX_RETRIES", "1")
    upstream([FakeResponse(500, text="internal")])
    response = client.post("/generate", json={"prompt": "anything"})
    assert response.status_code == 502
    assert response.get_json()["status"] == "error"


def test_generate_returns_504_on_timeout(client, upstream, no_sleep, monkeypatch):
    monkeypatch.setenv("HF_MAX_RETRIES", "1")
    upstream([requests.exceptions.Timeout("slow")])
    response = client.post("/generate", json={"prompt": "anything"})
    assert response.status_code == 504
    assert response.get_json()["error"] == "upstream_timeout"


def test_generate_passes_through_plain_text_scripts(client, upstream):
    upstream([chat_response("The hallway was longer on the way back.")])
    body = client.post("/generate", json={"prompt": "hallway"}).get_json()
    assert body["status"] == "success"
    assert body["script"]["format"] == "text"
    assert body["script"]["narration"].startswith("The hallway")


def test_unknown_route_returns_json_404(client):
    response = client.get("/nope")
    assert response.status_code == 404
    assert response.get_json()["error"] == "not_found"


def test_generate_rejects_get(client):
    response = client.get("/generate")
    assert response.status_code == 405
    assert response.get_json()["error"] == "method_not_allowed"
