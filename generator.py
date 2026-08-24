"""Turns a one-line idea into a production-ready YouTube horror script.

Talks to the Hugging Face Inference Providers router, which speaks the
OpenAI chat-completions dialect. The older `api-inference.huggingface.co`
serverless endpoint this project used to call has been retired, which is why
requests against it never produced a script.
"""

import json
import logging
import random
import re
import time

import requests

from config import config

log = logging.getLogger(__name__)

# Narration pace used to turn a target runtime into a word budget.
WORDS_PER_MINUTE = 140

SYSTEM_PROMPT = (
    "You are a veteran horror screenwriter who writes narration for YouTube "
    "videos. You write in a calm, patient, deeply unsettling voice: concrete "
    "sensory detail, ordinary places turned wrong, dread that builds through "
    "implication rather than gore. You never break character and never "
    "explain the craft behind the story."
)

# Retried: rate limits, cold models, and transient gateway failures.
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class ScriptGenerationError(Exception):
    """A failure that maps cleanly onto an HTTP response."""

    status_code = 502
    error_code = "generation_failed"

    def __init__(self, message, details=None):
        super().__init__(message)
        self.message = message
        self.details = details

    def to_dict(self):
        payload = {"status": "error", "error": self.error_code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload


class ConfigurationError(ScriptGenerationError):
    status_code = 503
    error_code = "not_configured"


class InvalidRequestError(ScriptGenerationError):
    status_code = 400
    error_code = "invalid_request"


class UpstreamError(ScriptGenerationError):
    status_code = 502
    error_code = "upstream_error"


class UpstreamTimeout(ScriptGenerationError):
    status_code = 504
    error_code = "upstream_timeout"


class ScriptBrief:
    """Validated description of the script the caller wants."""

    def __init__(self, prompt=None, tone=None, audience=None, duration_minutes=None,
                 section_count=None, temperature=None):
        self.prompt = self._clean_prompt(prompt)
        self.tone = self._clean_text(tone, "tone", 200) or "calm, eerie, unsettling"
        self.audience = self._clean_text(audience, "audience", 200)
        self.duration_minutes = self._clean_duration(duration_minutes)
        self.section_count = self._clean_section_count(section_count)
        self.temperature = self._clean_temperature(temperature)

    @staticmethod
    def _clean_prompt(value):
        if value is None:
            return "A creepy horror story set somewhere ordinary."
        if not isinstance(value, str):
            raise InvalidRequestError("'prompt' must be a string.")
        value = value.strip()
        if not value:
            return "A creepy horror story set somewhere ordinary."
        if len(value) > config.max_prompt_chars:
            raise InvalidRequestError(
                "'prompt' is too long (max %d characters)." % config.max_prompt_chars
            )
        return value

    @staticmethod
    def _clean_text(value, field, limit):
        if value is None:
            return None
        if not isinstance(value, str):
            raise InvalidRequestError("'%s' must be a string." % field)
        value = value.strip()
        if not value:
            return None
        if len(value) > limit:
            raise InvalidRequestError("'%s' is too long (max %d characters)." % (field, limit))
        return value

    @staticmethod
    def _clean_duration(value):
        if value is None:
            return 10
        try:
            minutes = int(value)
        except (TypeError, ValueError):
            raise InvalidRequestError("'duration_minutes' must be a number.")
        if not 1 <= minutes <= 60:
            raise InvalidRequestError("'duration_minutes' must be between 1 and 60.")
        return minutes

    @staticmethod
    def _clean_section_count(value):
        if value is None:
            return None
        try:
            count = int(value)
        except (TypeError, ValueError):
            raise InvalidRequestError("'section_count' must be a number.")
        if not 1 <= count <= 20:
            raise InvalidRequestError("'section_count' must be between 1 and 20.")
        return count

    @staticmethod
    def _clean_temperature(value):
        if value is None:
            return None
        try:
            temperature = float(value)
        except (TypeError, ValueError):
            raise InvalidRequestError("'temperature' must be a number.")
        if not 0.0 <= temperature <= 2.0:
            raise InvalidRequestError("'temperature' must be between 0.0 and 2.0.")
        return temperature

    @property
    def word_target(self):
        return self.duration_minutes * WORDS_PER_MINUTE

    @property
    def sections(self):
        if self.section_count:
            return self.section_count
        # Roughly one beat every two and a half minutes, kept in a sane range.
        return max(3, min(10, round(self.duration_minutes / 2.5)))


def build_user_prompt(brief):
    """Render the brief into the instruction sent to the model."""
    lines = [
        "Write a complete YouTube horror narration script.",
        "",
        "STORY IDEA: %s" % brief.prompt,
        "TONE: %s" % brief.tone,
        "TARGET RUNTIME: about %d minutes (roughly %d words of narration)."
        % (brief.duration_minutes, brief.word_target),
        "STRUCTURE: %d sections, each one a distinct beat that escalates the dread."
        % brief.sections,
    ]
    if brief.audience:
        lines.append("AUDIENCE: %s" % brief.audience)

    lines += [
        "",
        "Respond with a single JSON object and nothing else. No commentary, no "
        "markdown fences. Use exactly this shape:",
        "{",
        '  "title": "clickable YouTube title, under 70 characters",',
        '  "hook": "the first 15 seconds of narration, written to stop the scroll",',
        '  "sections": [',
        '    {"heading": "short beat name", "timestamp": "MM:SS", '
        '"narration": "the words the voice-over actually reads"}',
        "  ],",
        '  "outro": "closing narration, including the final unsettling image",',
        '  "description": "YouTube description, 2-3 sentences",',
        '  "tags": ["8-12 lowercase search tags"],',
        '  "thumbnail_ideas": ["2-3 concrete thumbnail concepts"]',
        "}",
        "",
        "The narration fields must contain finished prose to be read aloud, not "
        "stage directions or summaries. Timestamps must run in order from 00:00 "
        "and fit the target runtime.",
    ]
    return "\n".join(lines)


def _extract_json_object(text):
    """Pull a JSON object out of a model reply that may be wrapped in prose.

    Returns the parsed dict, or None if nothing usable is present.
    """
    if not text:
        return None

    # Strip ```json ... ``` fences the model may add despite instructions.
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    candidates = []
    if fenced:
        candidates.append(fenced.group(1))
    candidates.append(text)

    for candidate in candidates:
        candidate = candidate.strip()
        try:
            parsed = json.loads(candidate)
        except ValueError:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start == -1 or end <= start:
                continue
            try:
                parsed = json.loads(candidate[start:end + 1])
            except ValueError:
                continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _as_text_list(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [part.strip() for part in value.split(",") if part.strip()]
    return []


def _normalise_sections(raw_sections):
    sections = []
    if not isinstance(raw_sections, list):
        return sections
    for index, item in enumerate(raw_sections, start=1):
        if isinstance(item, str):
            sections.append({
                "heading": "Part %d" % index,
                "timestamp": None,
                "narration": item.strip(),
            })
            continue
        if not isinstance(item, dict):
            continue
        narration = item.get("narration") or item.get("text") or item.get("body") or ""
        sections.append({
            "heading": str(item.get("heading") or item.get("title") or "Part %d" % index).strip(),
            "timestamp": (str(item["timestamp"]).strip()
                          if item.get("timestamp") not in (None, "") else None),
            "narration": str(narration).strip(),
        })
    return sections


def _word_count(script):
    words = 0
    for field in ("hook", "outro"):
        words += len(str(script.get(field) or "").split())
    for section in script.get("sections") or []:
        words += len(str(section.get("narration") or "").split())
    return words


def parse_script(raw_text):
    """Turn the model's reply into the script payload we return to callers."""
    data = _extract_json_object(raw_text)

    if data is None:
        # The model ignored the JSON instruction. The prose is still the
        # product, so hand it back rather than failing the request.
        log.warning("Model reply was not JSON; returning it as plain text.")
        return {
            "format": "text",
            "title": None,
            "hook": None,
            "sections": [],
            "outro": None,
            "description": None,
            "tags": [],
            "thumbnail_ideas": [],
            "narration": raw_text.strip(),
            "word_count": len(raw_text.split()),
            "estimated_minutes": round(len(raw_text.split()) / WORDS_PER_MINUTE, 1),
        }

    script = {
        "format": "structured",
        "title": (str(data["title"]).strip() if data.get("title") else None),
        "hook": (str(data["hook"]).strip() if data.get("hook") else None),
        "sections": _normalise_sections(data.get("sections")),
        "outro": (str(data["outro"]).strip() if data.get("outro") else None),
        "description": (str(data["description"]).strip() if data.get("description") else None),
        "tags": _as_text_list(data.get("tags")),
        "thumbnail_ideas": _as_text_list(data.get("thumbnail_ideas")),
    }

    narration_parts = []
    if script["hook"]:
        narration_parts.append(script["hook"])
    narration_parts += [s["narration"] for s in script["sections"] if s["narration"]]
    if script["outro"]:
        narration_parts.append(script["outro"])
    script["narration"] = "\n\n".join(narration_parts)

    words = _word_count(script)
    script["word_count"] = words
    script["estimated_minutes"] = round(words / WORDS_PER_MINUTE, 1) if words else 0.0
    return script


def _build_payload(brief):
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(brief)},
        ],
        "max_tokens": config.max_tokens,
        "temperature": (brief.temperature if brief.temperature is not None
                        else config.temperature),
        "stream": False,
    }
    if config.json_mode:
        payload["response_format"] = {"type": "json_object"}
    return payload


def _extract_message_content(body):
    """Read the assistant text out of a chat-completions response body."""
    if not isinstance(body, dict):
        raise UpstreamError("Inference provider returned an unexpected response shape.")

    if "error" in body:
        error = body["error"]
        message = error.get("message") if isinstance(error, dict) else str(error)
        raise UpstreamError("Inference provider rejected the request.", details=message)

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise UpstreamError("Inference provider returned no choices.",
                            details=json.dumps(body)[:400])

    message = choices[0].get("message") or {}
    content = message.get("content")

    # Some providers stream-shape the first choice even for non-streaming calls.
    if content is None:
        content = (choices[0].get("delta") or {}).get("content")
    if content is None:
        content = choices[0].get("text")

    if isinstance(content, list):
        # Multi-part content blocks: keep the text parts.
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))

    if not isinstance(content, str) or not content.strip():
        raise UpstreamError("Inference provider returned an empty script.",
                            details=json.dumps(body)[:400])
    return content


def _sleep_for_attempt(attempt, retry_after=None):
    if retry_after is not None:
        delay = retry_after
    else:
        delay = config.retry_backoff * (2 ** attempt)
    delay += random.uniform(0, 0.5)  # jitter, so parallel callers do not sync up
    delay = min(delay, 30.0)
    log.info("Retrying inference request in %.1fs (attempt %d).", delay, attempt + 1)
    time.sleep(delay)


def _parse_retry_after(response):
    raw = response.headers.get("Retry-After") if response is not None else None
    if not raw:
        return None
    try:
        return max(0.0, min(float(raw), 30.0))
    except ValueError:
        return None


def call_model(brief, session=None):
    """POST the brief to the inference provider, retrying transient failures."""
    api_key = config.api_key
    if not api_key:
        raise ConfigurationError(
            "HF_API_KEY is not set, so scripts cannot be generated.",
            details="Set HF_API_KEY (or HF_TOKEN) to a Hugging Face access token "
                    "with inference permissions and restart the service.",
        )

    url = "%s/chat/completions" % config.api_base
    headers = {
        "Authorization": "Bearer %s" % api_key,
        "Content-Type": "application/json",
    }
    payload = _build_payload(brief)
    http = session or requests
    attempts = config.max_retries
    last_error = None

    for attempt in range(attempts):
        try:
            response = http.post(url, headers=headers, json=payload,
                                 timeout=config.request_timeout)
        except requests.exceptions.Timeout as exc:
            last_error = UpstreamTimeout(
                "The inference provider did not respond in time.", details=str(exc))
        except requests.exceptions.RequestException as exc:
            last_error = UpstreamError(
                "Could not reach the inference provider.", details=str(exc))
        else:
            status = response.status_code
            if status == 401 or status == 403:
                raise ConfigurationError(
                    "The inference provider rejected the API key.",
                    details=response.text[:400],
                )
            if status == 404:
                raise UpstreamError(
                    "Model '%s' is not available at %s." % (config.model, config.api_base),
                    details="Pick a model served by Hugging Face Inference Providers "
                            "and set HF_MODEL to it.",
                )
            if status in RETRYABLE_STATUS:
                last_error = UpstreamError(
                    "Inference provider is unavailable (HTTP %d)." % status,
                    details=response.text[:400],
                )
                if attempt < attempts - 1:
                    _sleep_for_attempt(attempt, _parse_retry_after(response))
                    continue
                raise last_error
            if status >= 400:
                raise UpstreamError(
                    "Inference provider returned HTTP %d." % status,
                    details=response.text[:400],
                )

            try:
                body = response.json()
            except ValueError:
                raise UpstreamError(
                    "Inference provider returned a non-JSON response.",
                    details=response.text[:400],
                )
            return _extract_message_content(body)

        # Network-level failure: back off and try again instead of bailing out.
        if attempt < attempts - 1:
            _sleep_for_attempt(attempt)

    raise last_error or UpstreamError("The inference provider could not be reached.")


def generate_script(brief, session=None):
    """Generate a script for `brief` and return the response payload."""
    raw_text = call_model(brief, session=session)
    script = parse_script(raw_text)
    return {
        "status": "success",
        "model": config.model,
        "request": {
            "prompt": brief.prompt,
            "tone": brief.tone,
            "audience": brief.audience,
            "duration_minutes": brief.duration_minutes,
            "sections": brief.sections,
        },
        "script": script,
    }
