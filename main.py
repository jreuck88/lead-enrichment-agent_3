import os
import json
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Initialize OpenAI client using your Render env var OPENAI_API_KEY
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return "✅ Lead Enrichment Agent is live."

@app.route("/enrich", methods=["POST"])
def enrich():
    try:
        data    = request.get_json(force=True)
        company = data.get("company_name")
        website = data.get("website")

        if not company or not website:
            return jsonify({"error": "Missing 'company_name' or 'website'"}), 400

        prompt = f"""
You are a business research assistant. Given a company name and website, return enriched lead data in JSON format.
Use only real, verifiable, public data. Do not make anything up.

Company: {company}
Website: {website}

Return ONLY this JSON object with these keys:
{{
  "CompanyName": "",
  "CompanyEmail": "",
  "Location": "",
  "BestPOC": "",
  "POCEmail": "",
  "POCLinkedIn": "",
  "InstagramURL": "",
  "LinkedInURL": "",
  "Website": "{website}",
  "CompanyServices": "",
  "ValueProp": "",
  "CompanySize": "",
  "AnnualRevenue": "",
  "LeadScore": 0
}}
"""
        # call GPT
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role":"system", "content":"You are a professional research assistant."},
                {"role":"user",   "content":prompt}
            ],
            temperature=0.3
        )
        content = resp.choices[0].message.content.strip()
        # strip any ``` fences
        if content.startswith("```json"):
            content = content.replace("```json","").replace("```","").strip()
        elif content.startswith("```"):
            content = content.replace("```","").strip()

        parsed = json.loads(content)
        return jsonify(parsed)

    except Exception as e:
        print("❌ /enrich error:", e)
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    try:
        data = request.get_json(force=True)
        b64  = data.get("image_base64")
        if not b64:
            return jsonify({"error": "Missing 'image_base64'"}), 400

        prompt = f"""
You are an OCR assistant. Given a base64-encoded image, extract the brand or company name and its official website.
Return ONLY a JSON object with keys `brandName` and `brandWebsite`.

Image (base64): {b64}
"""
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role":"system", "content": "You are an OCR assistant."},
                {"role":"user",   "content": prompt}
            ],
            temperature=0
        )
        content = resp.choices[0].message.content.strip()
        # strip backticks if any
        content = content.replace("```json","").replace("```","").strip()
        parsed  = json.loads(content)
        return jsonify(parsed)

    except Exception as e:
        print("❌ /analyze-image error:", e)
        return jsonify({"error": f"Server error: {e}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
