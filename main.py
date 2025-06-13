from flask import Flask, request, jsonify
import os
from openai import OpenAI

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/")
def home():
    return "✅ Lead Enrichment Agent is live."

@app.route("/enrich", methods=["POST"])
def enrich():
    data = request.get_json()
    company = data.get("company_name")
    website = data.get("website")

    if not company or not website:
        return jsonify({"error": "Missing company_name or website"}), 400

    prompt = f"""You are a business research assistant...
    Company: {company}
    Website: {website}
    Return JSON with fields:
    {{
      "CompanyName": "",
      ...
    }}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                { "role": "system", "content": "You are a professional assistant." },
                { "role": "user", "content": prompt }
            ],
            temperature=0.3,
        )
        content = response.choices[0].message.content.strip()

        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()

        return jsonify(json.loads(content))

    except Exception as e:
        return jsonify({"error": str(e)}), 500
