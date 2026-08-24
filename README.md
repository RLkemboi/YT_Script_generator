# YouTube Horror Script Generator

A small Flask API that turns a one-line story idea into a ready-to-record
YouTube horror script: title, hook, timestamped narration beats, outro,
description, tags, and thumbnail concepts.

Generation runs on [Claude](https://www.anthropic.com/claude) via the
official Anthropic SDK, using `claude-haiku-4-5` by default.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then put your Anthropic API key in it
python main.py              # loads .env automatically; http://localhost:5000
```

You need an Anthropic API key from
<https://console.anthropic.com/settings/keys>.

## API

### `GET /`

Service metadata, including whether an API key is configured.

### `GET /healthz`

Liveness probe. Always returns `200` while the process is up; the
`configured` field reports whether a key is present. Deploy platforms restart
services whose health check fails, so a missing key is reported rather than
signalled with a non-200.

### `POST /generate`

```bash
curl -X POST http://localhost:5000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "a neighbour who never blinks", "duration_minutes": 8}'
```

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `prompt` | string | a generic horror premise | The story idea. |
| `tone` | string | `calm, eerie, unsettling` | Narration voice. |
| `audience` | string | — | e.g. `true-crime listeners`. |
| `duration_minutes` | int 1–60 | `10` | Sets the word budget (~140 wpm) and the output token budget. |
| `section_count` | int 1–20 | derived from runtime | Number of story beats. |
| `temperature` | float 0–1 | `0.85` | Overrides `CLAUDE_TEMPERATURE`. Anthropic's range is 0–1. |

Successful response:

```json
{
  "status": "success",
  "model": "claude-haiku-4-5",
  "request": { "prompt": "...", "duration_minutes": 8, "sections": 3 },
  "script": {
    "format": "structured",
    "title": "The Neighbour Who Never Blinks",
    "hook": "I have lived next to him for six years...",
    "sections": [
      { "heading": "The Window", "timestamp": "00:20", "narration": "..." }
    ],
    "outro": "...",
    "description": "...",
    "tags": ["horror", "creepy story"],
    "thumbnail_ideas": ["..."],
    "narration": "hook + every section + outro, ready to read aloud",
    "word_count": 1120,
    "estimated_minutes": 8.0
  }
}
```

If the model ignores the JSON instruction, the prose is still returned, with
`"format": "text"` and the whole reply under `narration`.

### Error responses

Every error carries a real HTTP status and a stable `error` code:

| Status | `error` | Meaning |
| --- | --- | --- |
| 400 | `invalid_request` | Bad field in the request body. |
| 404 | `not_found` | No such endpoint. |
| 405 | `method_not_allowed` | Wrong HTTP method. |
| 502 | `upstream_error` | Claude failed, refused, or returned nothing usable. |
| 503 | `not_configured` | No API key, or Claude rejected it. |
| 504 | `upstream_timeout` | Claude did not respond in time. |

## Configuration

All settings are environment variables; see `.env.example`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | — | Anthropic API key. Required. |
| `CLAUDE_MODEL` | `claude-haiku-4-5` | Any Claude model your key can access. |
| `CLAUDE_TIMEOUT` | `180` | Per-request timeout, seconds. Keep below the Procfile's gunicorn `--timeout`. |
| `CLAUDE_MAX_RETRIES` | `2` | Passed to the Anthropic client, which retries 408/409/429/5xx and connection errors itself. |
| `CLAUDE_MAX_TOKENS_CEILING` | `16000` | Upper bound on the output budget, regardless of `duration_minutes`. |
| `CLAUDE_TEMPERATURE` | `0.85` | Sampling temperature (0–1). |
| `MAX_PROMPT_CHARS` | `4000` | Request prompt length cap. |
| `PORT` | `5000` | Bind port. |
| `LOG_LEVEL` | `INFO` | Logging verbosity. |

The output token budget scales with `duration_minutes` (roughly 1.6 tokens
per word of narration, plus overhead for the JSON structure), clamped
between 1024 and `CLAUDE_MAX_TOKENS_CEILING`.

## Deploying

This is a plain Flask + gunicorn app — any host that runs a persistent
Python process works. **Render** or **Fly.io** are recommended over a
serverless platform like Vercel or Netlify: their free-tier function
timeouts (roughly 10–60 seconds) are too short for a multi-minute script,
whereas a script can legitimately take over a minute to generate. Render's
free tier does cold-start after ~15 minutes idle; Fly's free allowance
stays warm.

The `Procfile` binds gunicorn to `$PORT`, which both platforms require:

```
web: gunicorn main:app --bind 0.0.0.0:${PORT:-5000} --workers 2 --threads 4 --timeout 240 --access-logfile -
```

**Render:** connect the repo, it detects the `Procfile` automatically. Add
`ANTHROPIC_API_KEY` under the service's Environment settings.

**Fly.io:** run `fly launch` (it generates a `fly.toml` from the
`Procfile`), then `fly secrets set ANTHROPIC_API_KEY=sk-ant-...`.

The gunicorn worker timeout (240s) is kept above `CLAUDE_TIMEOUT` (180s) so
a slow generation gets a clean error response instead of the worker being
killed mid-request. If you raise `duration_minutes` usage toward the 60
minute ceiling, raise both together.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite covers request validation, response parsing (including replies
wrapped in markdown fences or prose), and every HTTP status the API
returns, including how each Anthropic SDK exception maps onto one. No
network access or API key is needed — the Anthropic client is stubbed with
real SDK exception types so the mapping is tested against the actual
classes, not a guess at their shape.

## Notes on the model

`CLAUDE_MODEL` defaults to `claude-haiku-4-5`, Anthropic's fastest and
cheapest current model — a good fit for a high-volume, latency-sensitive
generation task like this. Set `CLAUDE_MODEL=claude-sonnet-5` or
`claude-opus-5` for higher-quality prose at a higher per-script cost.
