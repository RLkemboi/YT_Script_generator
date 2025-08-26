import os
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/", methods=["GET"])
def home():
    return "👻 Horror Script Generator is running!"

if __name__ == "__main__":
    app.run(debug=True)

@app.route("/generate", methods=["POST"])
def generate_horror_script():
    data = request.json
    user_prompt = data.get("prompt", "Write a creepy horror story.")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert horror storyteller. Write in a calm, eerie, unsettling tone."},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,
            max_tokens=800
        )
        story = response.choices[0].message.content.strip()
        return jsonify({"script": story})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
