"""HTTP entry point for the YouTube horror script generator.

`main:app` is the WSGI callable referenced by the Procfile.
"""

import logging
import os

from flask import Flask, jsonify, request

try:  # Load a local .env during development; absent in production images.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is a convenience, not a requirement
    pass

from config import config
from generator import (
    InvalidRequestError,
    ScriptBrief,
    ScriptGenerationError,
    generate_script,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("yt_script_generator")

app = Flask(__name__)


@app.get("/")
def home():
    return jsonify({
        "service": "YouTube horror script generator",
        "status": "ok",
        "configured": bool(config.api_key),
        "model": config.model,
        "endpoints": {
            "POST /generate": "generate a horror script from a story idea",
            "GET /healthz": "liveness and configuration check",
        },
    })


@app.get("/healthz")
def healthz():
    """Liveness check. Reports configuration without failing the probe.

    Deploy platforms restart a service whose health check fails, so a missing
    API key is reported here rather than turned into a non-200 response.
    """
    return jsonify({
        "status": "ok",
        "configured": bool(config.api_key),
        "model": config.model,
        "api_base": config.api_base,
    })


@app.post("/generate")
def generate():
    data = request.get_json(force=True, silent=True)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        return jsonify(InvalidRequestError(
            "Request body must be a JSON object.").to_dict()), 400

    try:
        brief = ScriptBrief(
            prompt=data.get("prompt"),
            tone=data.get("tone"),
            audience=data.get("audience"),
            duration_minutes=data.get("duration_minutes"),
            section_count=data.get("section_count"),
            temperature=data.get("temperature"),
        )
        result = generate_script(brief)
    except ScriptGenerationError as exc:
        log.warning("Script generation failed: %s (%s)", exc.message, exc.details)
        return jsonify(exc.to_dict()), exc.status_code

    log.info("Generated a %d-word script with %d sections.",
             result["script"]["word_count"], len(result["script"]["sections"]))
    return jsonify(result), 200


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"status": "error", "error": "not_found",
                    "message": "No such endpoint."}), 404


@app.errorhandler(405)
def method_not_allowed(_error):
    return jsonify({"status": "error", "error": "method_not_allowed",
                    "message": "That endpoint does not accept this HTTP method."}), 405


@app.errorhandler(500)
def internal_error(error):
    log.exception("Unhandled error: %s", error)
    return jsonify({"status": "error", "error": "internal_error",
                    "message": "Something went wrong generating the script."}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "").lower() in ("1", "true", "yes")
    app.run(host="0.0.0.0", port=port, debug=debug)
