import os
import time
import requests
from flask import Flask, request, jsonify

# Load Hugging Face API Key
HF_API_KEY = os.getenv("HF_API_KEY")
print("HF_API_KEY loaded:", "✅" if HF_API_KEY else "❌")

if HF_API_KEY is None:
    raise ValueError("HF_API_KEY is not set in environment variables")

app = Flask(__name__)

# Hugging Face model (you can swap)
HF_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

@app.route("/", methods=["GET"])
def home():
    return "👻 Horror Script Generator is running (Hugging Face)!"

@app.route("/generate", methods=["POST"])
def generate_horror_script():
    # --- 1. Read user input safely ---
    data = request.get_json(force=True, silent=True) or {}
    user_prompt = data.get("prompt", "Write a creepy horror story.")

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": f"You are an expert horror storyteller. Write in a calm, eerie, unsettling tone.\n\n{user_prompt}",
        "parameters": {"max_new_tokens": 800, "temperature": 0.8}
    }

    # --- 2. Retry logic if HF model is still loading ---
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{HF_MODEL}",
                headers=headers,
                json=payload,
                timeout=60
            )

            print("HF Status:", response.status_code)
            print("HF Raw Response:", response.text[:300])  # first 300 chars for logs

            # Try to parse JSON
            try:
                result = response.json()
            except Exception:
                return jsonify({
                    "status": "error",
                    "details": "Invalid JSON from Hugging Face",
                    "raw": response.text
                }), 200

            # --- 3. Handle Hugging Face API errors ---
            if isinstance(result, dict) and "error" in result:
                # If model is still loading, retry a few times
                if "loading" in result["error"].lower() and attempt < max_retries - 1:
                    print("⏳ Model still loading... retrying")
                    time.sleep(10)  # wait 10s before retry
                    continue
                return jsonify({
                    "status": "huggingface_error",
                    "details": result["error"]
                }), 200

            # --- 4. Extract generated text ---
            story = None
            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    story = result[0]["generated_text"]

            if story is None:  # fallback
                story = result.get("generated_text") if isinstance(result, dict) else str(result)

            return jsonify({
                "status": "success",
                "script": story
            }), 200

        except Exception as e:
            # Handle network or unexpected errors
            return jsonify({
                "status": "error",
                "details": str(e)
            }), 200

    # --- 5. If all retries fail ---
    return jsonify({
        "status": "failed",
        "details": "Model did not respond after retries"
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
