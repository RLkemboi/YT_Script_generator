# YouTube Horror Script Generator

A small Flask API that turns a one-line story idea into a ready-to-record
YouTube horror script: title, hook, timestamped narration beats, outro,
description, tags, and thumbnail concepts.

Generation runs on [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers).

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then put your Hugging Face token in it
python main.py              # loads .env automatically; http://localhost:5000
```

You need a Hugging Face access token with the **"Make calls to Inference
Providers"** permission, from <https://huggingface.co/settings/tokens>.

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
| `duration_minutes` | int 1–60 | `10` | Sets the word budget (~140 wpm). |
| `section_count` | int 1–20 | derived from runtime | Number of story beats. |
| `temperature` | float 0–2 | `0.85` | Overrides `HF_TEMPERATURE`. |

Successful response:

```json
{
  "status": "success",
  "model": "Qwen/Qwen2.5-72B-Instruct",
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
| 502 | `upstream_error` | The inference provider failed or returned nothing usable. |
| 503 | `not_configured` | No API key, or the provider rejected it. |
| 504 | `upstream_timeout` | The provider did not respond in time. |

## Configuration

All settings are environment variables; see `.env.example`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `HF_API_KEY` / `HF_TOKEN` | — | Hugging Face access token. Required. |
| `HF_MODEL` | `Qwen/Qwen2.5-72B-Instruct` | Any model served by Inference Providers. |
| `HF_API_BASE` | `https://router.huggingface.co/v1` | OpenAI-compatible chat-completions base URL. |
| `HF_TIMEOUT` | `90` | Per-request timeout, seconds. |
| `HF_MAX_RETRIES` | `3` | Attempts before giving up. |
| `HF_RETRY_BACKOFF` | `2` | First retry delay; doubles each attempt, honours `Retry-After`. |
| `HF_MAX_TOKENS` | `2000` | Output cap. Raise for scripts over ~12 minutes. |
| `HF_TEMPERATURE` | `0.85` | Sampling temperature. |
| `HF_JSON_MODE` | `false` | Send `response_format: json_object`. Not every provider supports it. |
| `MAX_PROMPT_CHARS` | `4000` | Request prompt length cap. |
| `PORT` | `5000` | Bind port. |
| `LOG_LEVEL` | `INFO` | Logging verbosity. |

Because `HF_API_BASE` is just an OpenAI-compatible endpoint, pointing it at
another gateway is enough to switch providers.

## Deploying

The `Procfile` binds gunicorn to `$PORT`, which is what Heroku, Render, and
Railway require:

```
web: gunicorn main:app --bind 0.0.0.0:${PORT:-5000} --workers 2 --threads 4 --timeout 180 --access-logfile -
```

The timeout is deliberately long: a 10-minute script can take a provider well
over a minute to produce. Set `HF_API_KEY` in the platform's environment
settings.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite covers request validation, response parsing (including replies
wrapped in markdown fences or prose), retry and backoff behaviour, and every
HTTP status the API returns. No network access or API key is needed — the
inference provider is stubbed.

## Notes on the model

`HF_MODEL` defaults to an openly licensed instruct model that Inference
Providers serves without a gated-access request. Larger or gated models
(Llama, Mixtral) work too, but you must accept their terms on the Hugging Face
model page first, or requests come back `404`.
