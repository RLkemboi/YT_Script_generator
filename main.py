import os
import requests
from flask import Flask, request, jsonify

# Load Hugging Face API key from environment variables
HF_API_KEY = os.getenv("HF_API_KEY")

if not HF_API_KEY:
    raise ValueError("❌ HF_API_KEY is not set in environment variables on Render.")

# Flask app
app = Flask(__name__)

# Model to use
HF_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # can swap to another model

@app.route("/", methods=["GET"])
def home():
    return "👻 Horror Script Generator is running (Hugging Face)!"

@app.route("/generate", methods=["POST"])
def generate_horror_script():
    data = request.json or {}
    user_prompt = data.get("prompt", "Write a creepy horror story.")

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": f"You are an expert horror storyteller. Write in a calm, eerie, unsettling tone.\n\n{user_prompt}",
        "parameters": {"max_new_tokens": 800, "temperature": 0.8}
    }

    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL}",
            headers=headers,
            json=payload,
            timeout=60
        )
        result = response.json()

        # Debug log to check full Hugging Face response in Render logs
        print("🔎 HF response:", result, flush=True)

        if "error" in result:
            return jsonify({"error": result["error"]}), 500

        # Hugging Face responses can vary depending on the model
        if isinstance(result, list) and "generated_text" in result[0]:
            story = result[0]["generated_text"]
        elif "generated_text" in result:
            story = result["generated_text"]
        else:
            story = str(result)

        return jsonify({"script": story})

    except Exception as e:
        print("❌ Exception:", str(e), flush=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Important: Render binds to 0.0.0.0 and uses PORT env variable
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
