import json

import anthropic
import pytest

import generator
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
    assert brief.max_tokens == int(1400 * 1.6) + 400
    assert "horror" in brief.prompt.lower()


def test_brief_accepts_overrides():
    brief = ScriptBrief(prompt="  A lighthouse keeper  ", duration_minutes=20,
                        section_count=6, tone="bleak", audience="adults")
    assert brief.prompt == "A lighthouse keeper"
    assert brief.sections == 6
    assert brief.word_target == 2800
    assert brief.tone == "bleak"


def test_brief_max_tokens_respects_ceiling(monkeypatch):
    monkeypatch.setenv("CLAUDE_MAX_TOKENS_CEILING", "3000")
    brief = ScriptBrief(duration_minutes=60)  # would estimate well above the ceiling
    assert brief.max_tokens == 3000


def test_brief_max_tokens_has_a_floor(monkeypatch):
    monkeypatch.setenv("CLAUDE_MAX_TOKENS_CEILING", "500")  # below MIN_MAX_TOKENS
    brief = ScriptBrief(duration_minutes=1)
    assert brief.max_tokens == generator.MIN_MAX_TOKENS


@pytest.mark.parametrize("kwargs", [
    {"prompt": 123},
    {"duration_minutes": 0},
    {"duration_minutes": 61},
    {"duration_minutes": "soon"},
    {"section_count": 0},
    {"section_count": 99},
    {"temperature": 1.5},
    {"temperature": -1},
])
def test_brief_rejects_bad_input(kwargs):
    with pytest.raises(InvalidRequestError):
        ScriptBrief(**kwargs)


def test_brief_accepts_temperature_within_anthropic_range():
    assert ScriptBrief(temperature=1.0).temperature == 1.0
    assert ScriptBrief(temperature=0.0).temperature == 0.0


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

def test_call_model_sends_expected_request(fake_client, make_chat_message):
    client = fake_client([make_chat_message("hello")])
    brief = ScriptBrief()
    assert call_model(brief, client=client) == "hello"
    call = client.calls[0]
    assert call["model"] == "claude-haiku-4-5"
    assert call["system"] == generator.SYSTEM_PROMPT
    assert call["messages"] == [
        {"role": "user", "content": generator.build_user_prompt(brief)}
    ]
    assert call["max_tokens"] == brief.max_tokens
    assert call["temperature"] == 0.85


def test_call_model_respects_model_override(monkeypatch, fake_client, make_chat_message):
    monkeypatch.setenv("CLAUDE_MODEL", "claude-opus-5")
    client = fake_client([make_chat_message("hi")])
    call_model(ScriptBrief(), client=client)
    assert client.calls[0]["model"] == "claude-opus-5"


def test_call_model_uses_brief_temperature_override(fake_client, make_chat_message):
    client = fake_client([make_chat_message("hi")])
    call_model(ScriptBrief(temperature=0.2), client=client)
    assert client.calls[0]["temperature"] == 0.2


def test_call_model_requires_api_key(monkeypatch, fake_client):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ConfigurationError) as excinfo:
        call_model(ScriptBrief(), client=fake_client([]))
    assert excinfo.value.status_code == 503


@pytest.mark.parametrize("error_class,status", [
    (anthropic.AuthenticationError, 401),
    (anthropic.PermissionDeniedError, 403),
])
def test_call_model_maps_bad_key_errors(fake_client, anthropic_error, error_class, status):
    client = fake_client([anthropic_error(error_class, status)])
    with pytest.raises(ConfigurationError):
        call_model(ScriptBrief(), client=client)


def test_call_model_maps_not_found_to_upstream_error(fake_client, anthropic_error):
    client = fake_client([anthropic_error(anthropic.NotFoundError, 404)])
    with pytest.raises(UpstreamError) as excinfo:
        call_model(ScriptBrief(), client=client)
    assert "CLAUDE_MODEL" in excinfo.value.details


def test_call_model_maps_rate_limit(fake_client, anthropic_error):
    client = fake_client([anthropic_error(anthropic.RateLimitError, 429)])
    with pytest.raises(UpstreamError):
        call_model(ScriptBrief(), client=client)


def test_call_model_maps_generic_status_error(fake_client, anthropic_error):
    client = fake_client([anthropic_error(anthropic.APIStatusError, 500)])
    with pytest.raises(UpstreamError):
        call_model(ScriptBrief(), client=client)


def test_call_model_maps_timeout(fake_client, anthropic_error):
    client = fake_client([anthropic_error(anthropic.APITimeoutError)])
    with pytest.raises(UpstreamTimeout) as excinfo:
        call_model(ScriptBrief(), client=client)
    assert excinfo.value.status_code == 504


def test_call_model_maps_connection_error(fake_client, anthropic_error):
    client = fake_client([anthropic_error(anthropic.APIConnectionError)])
    with pytest.raises(UpstreamError):
        call_model(ScriptBrief(), client=client)


def test_call_model_raises_on_refusal(fake_client, make_chat_message, fake_stop_details_cls):
    msg = make_chat_message(text=None, stop_reason="refusal",
                            stop_details=fake_stop_details_cls(explanation="policy"))
    client = fake_client([msg])
    with pytest.raises(UpstreamError) as excinfo:
        call_model(ScriptBrief(), client=client)
    assert excinfo.value.details == "policy"


def test_call_model_rejects_empty_content(fake_client, make_chat_message):
    client = fake_client([make_chat_message("   ")])
    with pytest.raises(UpstreamError):
        call_model(ScriptBrief(), client=client)


def test_call_model_joins_multiple_text_blocks(fake_client, fake_message_cls, make_text_block):
    blocks = [make_text_block("part one "), make_text_block("part two")]
    client = fake_client([fake_message_cls(content=blocks)])
    assert call_model(ScriptBrief(), client=client) == "part one part two"


# --- end to end -------------------------------------------------------------

def test_generate_script_returns_full_payload(fake_client, make_chat_message):
    client = fake_client([make_chat_message(json.dumps(VALID_SCRIPT))])
    result = generate_script(ScriptBrief(prompt="A house that counts"), client=client)
    assert result["status"] == "success"
    assert result["request"]["prompt"] == "A house that counts"
    assert result["script"]["title"] == VALID_SCRIPT["title"]
    assert result["model"] == "claude-haiku-4-5"
