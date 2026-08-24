import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Give every test a predictable environment."""
    for name in ("HF_API_KEY", "HF_TOKEN", "HF_MODEL", "HF_API_BASE", "HF_JSON_MODE",
                 "HF_MAX_RETRIES", "HF_RETRY_BACKOFF", "HF_TIMEOUT", "HF_MAX_TOKENS",
                 "HF_TEMPERATURE", "MAX_PROMPT_CHARS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HF_API_KEY", "test-key")
    # Keep retry backoff from making the suite slow.
    monkeypatch.setenv("HF_RETRY_BACKOFF", "0")


@pytest.fixture
def no_sleep(monkeypatch):
    import generator
    monkeypatch.setattr(generator.time, "sleep", lambda _seconds: None)


class FakeResponse:
    def __init__(self, status_code=200, json_body=None, text=None, headers=None):
        self.status_code = status_code
        self._json_body = json_body
        self.text = text if text is not None else ""
        self.headers = headers or {}

    def json(self):
        if self._json_body is None:
            raise ValueError("no json")
        return self._json_body


class FakeSession:
    """Stands in for `requests`, replaying a queued list of outcomes."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        if not self.outcomes:
            raise AssertionError("FakeSession received an unexpected extra call")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def chat_response(content):
    return FakeResponse(200, {"choices": [{"message": {"role": "assistant",
                                                       "content": content}}]})


@pytest.fixture
def fake_session():
    return FakeSession


@pytest.fixture
def make_chat_response():
    return chat_response


@pytest.fixture
def fake_response():
    return FakeResponse
