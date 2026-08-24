import json

import pytest
import requests

import generator
from conftest import FakeResponse, FakeSession, chat_response
from generator import (
    ConfigurationError,
    InvalidRequestError,
    ScriptBrief,
    UpstreamError,
    UpstreamTimeout,
    call_model,
    generate_script,
    parse_script,
)

from tests_data import VALID_SCRIPT


# --- brief validation -------------------------------------------------------

def test_brief_defaults():
    brief = ScriptBrief()
    assert brief.duration_minutes == 10
    assert brief.word_target == 1400
    assert brief.sections == 4
    assert "horror" in brief.prompt.lower()


def test_brief_accepts_overrides():
    brief = ScriptBrief(prompt="  A lighthouse keeper  ", duration_minutes=20,
                        section_count=6, tone="bleak", audience="adults")
    assert brief.prompt == "A lighthouse keeper"
    assert brief.sections == 6
    assert brief.word_target == 2800
    assert brief.tone == "bleak"


@pytest.mark.parametrize("kwargs", [
    {"prompt": 123},
    {"duration_minutes": 0},
    {"duration_minutes": 61},
    {"duration_minutes": "soon"},
    {"section_count": 0},
    {"section_count": 99},
    {"temperature": 5},
    {"temperature": -1},
])
def test_brief_rejects_bad_input(kwargs):
    with pytest.raises(InvalidRequestError):
        ScriptBrief(**kwargs)


def test_brief_rejects_overlong_prompt(monkeypatch):
    monkeypatch.setenv("MAX_PROMPT_CHARS", "10")
    with pytest.raises(InvalidRequestError):
        ScriptBrief(prompt="x" * 11)


# --- response parsing -------------------------------------------------------

def test_parse_structured_script():
    script = parse_script(json.dumps(VALID_SCRIPT))
    assert script["format"] == "structured"
    assert script["title"] == VALID_SCRIPT["title"]
    assert len(script["sections"]) == 2
    assert script["sections"][0]["timestamp"] == "00:15"
    assert script["tags"] == ["horror story", "creepy", "true scary"]
    assert script["word_count"] > 0
    assert "Fourteen creaks" in script["narration"]


def test_parse_strips_markdown_fences():
    raw = "Here you go!\n```json\n%s\n```\nHope that works." % json.dumps(VALID_SCRIPT)
    script = parse_script(raw)
    assert script["format"] == "structured"
    assert script["title"] == VALID_SCRIPT["title"]


def test_parse_recovers_json_embedded_in_prose():
    raw = "Sure thing. %s That's the script." % json.dumps(VALID_SCRIPT)
    script = parse_script(raw)
    assert script["format"] == "structured"
    assert len(script["sections"]) == 2


def test_parse_falls_back_to_plain_text():
    raw = "The door was open again. " * 20
    script = parse_script(raw)
    assert script["format"] == "text"
    assert script["narration"].startswith("The door was open")
    assert script["word_count"] == 100
    assert script["estimated_minutes"] > 0


def test_parse_handles_sections_as_bare_strings():
    script = parse_script(json.dumps({"title": "T", "sections": ["first beat", "second beat"]}))
    assert [s["heading"] for s in script["sections"]] == ["Part 1", "Part 2"]
    assert script["sections"][0]["narration"] == "first beat"


def test_parse_handles_comma_separated_tags():
    script = parse_script(json.dumps({"title": "T", "tags": "horror, creepy , scary"}))
    assert script["tags"] == ["horror", "creepy", "scary"]


def test_parse_tolerates_missing_fields():
    script = parse_script(json.dumps({"hook": "Just a hook."}))
    assert script["format"] == "structured"
    assert script["title"] is None
    assert script["sections"] == []
    assert script["tags"] == []


# --- upstream call ----------------------------------------------------------

def test_call_model_posts_to_chat_completions():
    session = FakeSession([chat_response("hello")])
    assert call_model(ScriptBrief(), session=session) == "hello"
    call = session.calls[0]
    assert call["url"] == "https://router.huggingface.co/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer test-key"
    assert call["json"]["messages"][0]["role"] == "system"
    assert "YouTube horror narration script" in call["json"]["messages"][1]["content"]
    assert call["json"]["stream"] is False


def test_call_model_respects_env_overrides(monkeypatch):
    monkeypatch.setenv("HF_API_BASE", "https://example.test/v1/")
    monkeypatch.setenv("HF_MODEL", "some/model")
    monkeypatch.setenv("HF_JSON_MODE", "true")
    session = FakeSession([chat_response("hi")])
    call_model(ScriptBrief(), session=session)
    call = session.calls[0]
    assert call["url"] == "https://example.test/v1/chat/completions"
    assert call["json"]["model"] == "some/model"
    assert call["json"]["response_format"] == {"type": "json_object"}


def test_call_model_requires_api_key(monkeypatch):
    monkeypatch.delenv("HF_API_KEY", raising=False)
    with pytest.raises(ConfigurationError) as excinfo:
        call_model(ScriptBrief(), session=FakeSession([]))
    assert excinfo.value.status_code == 503


def test_call_model_accepts_hf_token_alias(monkeypatch):
    monkeypatch.delenv("HF_API_KEY", raising=False)
    monkeypatch.setenv("HF_TOKEN", "alias-key")
    session = FakeSession([chat_response("ok")])
    call_model(ScriptBrief(), session=session)
    assert session.calls[0]["headers"]["Authorization"] == "Bearer alias-key"


@pytest.mark.parametrize("status", [401, 403])
def test_call_model_reports_bad_key(status):
    session = FakeSession([FakeResponse(status, text="invalid token")])
    with pytest.raises(ConfigurationError):
        call_model(ScriptBrief(), session=session)


def test_call_model_reports_unknown_model():
    session = FakeSession([FakeResponse(404, text="not found")])
    with pytest.raises(UpstreamError) as excinfo:
        call_model(ScriptBrief(), session=session)
    assert "HF_MODEL" in excinfo.value.details


def test_call_model_retries_on_503_then_succeeds(no_sleep):
    session = FakeSession([
        FakeResponse(503, text="model loading"),
        FakeResponse(503, text="model loading"),
        chat_response("finally"),
    ])
    assert call_model(ScriptBrief(), session=session) == "finally"
    assert len(session.calls) == 3


def test_call_model_retries_network_errors(no_sleep):
    """The old code returned on the first exception, so retries never ran."""
    session = FakeSession([
        requests.exceptions.ConnectionError("boom"),
        chat_response("recovered"),
    ])
    assert call_model(ScriptBrief(), session=session) == "recovered"
    assert len(session.calls) == 2


def test_call_model_gives_up_after_max_retries(no_sleep, monkeypatch):
    monkeypatch.setenv("HF_MAX_RETRIES", "2")
    session = FakeSession([FakeResponse(503, text="still loading")] * 2)
    with pytest.raises(UpstreamError):
        call_model(ScriptBrief(), session=session)
    assert len(session.calls) == 2


def test_call_model_maps_timeout(no_sleep, monkeypatch):
    monkeypatch.setenv("HF_MAX_RETRIES", "1")
    session = FakeSession([requests.exceptions.Timeout("too slow")])
    with pytest.raises(UpstreamTimeout) as excinfo:
        call_model(ScriptBrief(), session=session)
    assert excinfo.value.status_code == 504


def test_call_model_honours_retry_after_header(no_sleep, monkeypatch):
    seen = []
    monkeypatch.setattr(generator.time, "sleep", lambda s: seen.append(s))
    session = FakeSession([
        FakeResponse(429, text="slow down", headers={"Retry-After": "5"}),
        chat_response("ok"),
    ])
    call_model(ScriptBrief(), session=session)
    assert 5.0 <= seen[0] <= 5.5


def test_call_model_rejects_non_json_body():
    session = FakeSession([FakeResponse(200, json_body=None, text="<html>gateway</html>")])
    with pytest.raises(UpstreamError) as excinfo:
        call_model(ScriptBrief(), session=session)
    assert "non-JSON" in excinfo.value.message


def test_call_model_surfaces_provider_error_object():
    session = FakeSession([FakeResponse(200, {"error": {"message": "quota exceeded"}})])
    with pytest.raises(UpstreamError) as excinfo:
        call_model(ScriptBrief(), session=session)
    assert excinfo.value.details == "quota exceeded"


def test_call_model_rejects_empty_content():
    session = FakeSession([FakeResponse(200, {"choices": [{"message": {"content": "  "}}]})])
    with pytest.raises(UpstreamError):
        call_model(ScriptBrief(), session=session)


def test_call_model_handles_content_blocks():
    session = FakeSession([FakeResponse(200, {"choices": [{"message": {"content": [
        {"type": "text", "text": "part one "}, {"type": "text", "text": "part two"}]}}]})])
    assert call_model(ScriptBrief(), session=session) == "part one part two"


# --- end to end -------------------------------------------------------------

def test_generate_script_returns_full_payload():
    session = FakeSession([chat_response(json.dumps(VALID_SCRIPT))])
    result = generate_script(ScriptBrief(prompt="A house that counts"), session=session)
    assert result["status"] == "success"
    assert result["request"]["prompt"] == "A house that counts"
    assert result["script"]["title"] == VALID_SCRIPT["title"]
    assert result["model"] == "Qwen/Qwen2.5-72B-Instruct"
