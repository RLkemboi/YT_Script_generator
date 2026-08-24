import os
import sys

import anthropic
import httpx2
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Give every test a predictable environment."""
    for name in ("ANTHROPIC_API_KEY", "CLAUDE_MODEL", "CLAUDE_TIMEOUT",
                 "CLAUDE_MAX_RETRIES", "CLAUDE_MAX_TOKENS_CEILING",
                 "CLAUDE_TEMPERATURE", "MAX_PROMPT_CHARS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


class FakeStopDetails:
    def __init__(self, explanation=None, category=None):
        self.explanation = explanation
        self.category = category


class FakeMessage:
    """Stands in for `anthropic.types.Message`."""

    def __init__(self, content=(), stop_reason="end_turn", stop_details=None):
        self.content = list(content)
        self.stop_reason = stop_reason
        self.stop_details = stop_details


def chat_message(text=None, stop_reason="end_turn", stop_details=None):
    blocks = []
    if text is not None:
        blocks.append(anthropic.types.TextBlock(type="text", text=text, citations=None))
    return FakeMessage(content=blocks, stop_reason=stop_reason, stop_details=stop_details)


class FakeMessagesResource:
    def __init__(self, outcomes, calls):
        self._outcomes = outcomes
        self._calls = calls

    def create(self, **kwargs):
        self._calls.append(kwargs)
        if not self._outcomes:
            raise AssertionError("FakeClient received an unexpected extra call")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeClient:
    """Stands in for `anthropic.Anthropic`, replaying queued outcomes."""

    def __init__(self, outcomes):
        self.calls = []
        self.messages = FakeMessagesResource(list(outcomes), self.calls)


def _fake_request():
    return httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def _fake_response(status_code):
    return httpx2.Response(status_code, request=_fake_request())


def make_anthropic_error(error_class, status_code=None, message="error"):
    """Construct a real Anthropic SDK exception, the way the SDK itself would."""
    if error_class is anthropic.APIConnectionError:
        return error_class(message=message, request=_fake_request())
    if error_class is anthropic.APITimeoutError:
        return error_class(request=_fake_request())
    return error_class(message, response=_fake_response(status_code), body=None)


@pytest.fixture
def fake_client():
    return FakeClient


@pytest.fixture
def make_chat_message():
    return chat_message


@pytest.fixture
def make_text_block():
    return lambda text: anthropic.types.TextBlock(type="text", text=text, citations=None)


@pytest.fixture
def fake_message_cls():
    return FakeMessage


@pytest.fixture
def fake_stop_details_cls():
    return FakeStopDetails


@pytest.fixture
def anthropic_error():
    return make_anthropic_error
