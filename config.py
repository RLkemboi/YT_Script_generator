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


class Config:
    """Reads configuration from the environment on every access.

    Values are read lazily rather than at import time so that a missing API
    key degrades a single endpoint instead of preventing the app from booting.
    """

    @property
    def api_key(self):
        return os.getenv("ANTHROPIC_API_KEY")

    @property
    def model(self):
        return os.getenv("CLAUDE_MODEL", "claude-haiku-4-5")

    @property
    def request_timeout(self):
        """Per-request timeout, seconds. Keep below the Procfile's gunicorn
        --timeout so a slow generation gets a clean error instead of a
        SIGKILL mid-request."""
        return _env_float("CLAUDE_TIMEOUT", 180.0)

    @property
    def max_retries(self):
        """Passed straight to the Anthropic client; it handles the actual
        backoff for 408/409/429/5xx and connection errors."""
        return max(0, _env_int("CLAUDE_MAX_RETRIES", 2))

    @property
    def max_tokens_ceiling(self):
        """Upper bound on the output budget, regardless of duration_minutes."""
        return _env_int("CLAUDE_MAX_TOKENS_CEILING", 16000)

    @property
    def temperature(self):
        return _env_float("CLAUDE_TEMPERATURE", 0.85)

    @property
    def max_prompt_chars(self):
        return _env_int("MAX_PROMPT_CHARS", 4000)


config = Config()
