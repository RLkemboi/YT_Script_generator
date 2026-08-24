"""Environment-driven configuration for the horror script generator."""

import os


def _env_int(name, default):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name, default):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


class Config:
    """Reads configuration from the environment on every access.

    Values are read lazily rather than at import time so that a missing API
    key degrades a single endpoint instead of preventing the app from booting.
    """

    @property
    def api_key(self):
        # HF_TOKEN is the name the Hugging Face tooling uses; accept both.
        return os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN")

    @property
    def api_base(self):
        return os.getenv("HF_API_BASE", "https://router.huggingface.co/v1").rstrip("/")

    @property
    def model(self):
        return os.getenv("HF_MODEL", "Qwen/Qwen2.5-72B-Instruct")

    @property
    def request_timeout(self):
        return _env_float("HF_TIMEOUT", 90.0)

    @property
    def max_retries(self):
        return max(1, _env_int("HF_MAX_RETRIES", 3))

    @property
    def retry_backoff(self):
        """Seconds to wait before the first retry; doubles on each attempt."""
        return _env_float("HF_RETRY_BACKOFF", 2.0)

    @property
    def max_tokens(self):
        return _env_int("HF_MAX_TOKENS", 2000)

    @property
    def temperature(self):
        return _env_float("HF_TEMPERATURE", 0.85)

    @property
    def json_mode(self):
        """Ask the provider to constrain output to JSON.

        Off by default: not every Hugging Face inference provider supports
        `response_format`, and the prompt plus the tolerant parser already
        handle plain-text replies.
        """
        return _env_bool("HF_JSON_MODE", False)

    @property
    def max_prompt_chars(self):
        return _env_int("MAX_PROMPT_CHARS", 4000)


config = Config()
