import os
import requests
from flask import Flask, request, jsonify

HF_API_KEY = os.getenv("HF_API_KEY")
print("HF_API_KEY loaded:", "✅" if HF_API_KEY else "❌")

if HF_API_KEY is None:
    raise ValueError("HF_API_KEY is not set in environment variables")

app = Flask(__name__)

HF_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # you can swap this


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

        # ✅ Log status + raw response
        print("HF Status:", response.status_code)
        print("HF Raw Response:", response.text[:500])  # only show first 500 chars

        # Try JSON parsing safely
        try:
            result = response.json()
        except Exception as parse_err:
            return jsonify({
                "error": f"Invalid JSON from Hugging Face",
                "details": response.text
            }), 500

        # Handle HF API errors
        if "error" in result:
            return jsonify({"error": result["error"]}), 500

        # Handle text-generation response structure
        if isinstance(result, list) and "generated_text" in result[0]:
            story = result[0]["generated_text"]
        else:
            story = str(result)

        return jsonify({"script": story})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
