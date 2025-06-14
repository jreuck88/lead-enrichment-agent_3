import os
import json
import base64
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)

# Initialize OpenAI client using the OPENAI_API_KEY env var in Render
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/")
def home():
    return "✅ Lead Enrichment & Image OCR Agent is live."

@app.route("/enrich", methods=["POST"])
def enrich():
    try:
        data = request.get_json(force=True)
        company = data.get("company_name")
        website = data.get("website")

        if not company or not website:
            return jsonify({"error": "Missing 'company_name' or 'website'"}), 400

        prompt = f"""
You are a business research assistant. Given a company name and website, return enriched lead data in JSON format.
Use only real, verifiable, public data. Do not make up anything.

Company: {company}
Website: {website}

Return only this JSON object:
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

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional research assistant."},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.3
        )

        content = response.choices[0].message.content.strip()

        # strip any ``` fences
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()

        parsed = json.loads(content)
        return jsonify(parsed)

    except Exception as e:
        print("❌ Backend error:", e)
        return jsonify({"error": f"Server error: {e}"}), 500


@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    try:
        data = request.get_json(force=True)
        b64 = data.get("image_base64")
        if not b64:
            return jsonify({"error": "Missing 'image_base64'"}), 400

        # call OpenAI GPT-4 Vision (or equivalent) for OCR/brand extraction
        img_bytes = base64.b64decode(b64)
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or your available GPT-4V model
            messages=[
                {"role": "system", "content": "You are an OCR assistant. Extract the brand or company name from the given image."},
                {"role": "user",   "content": "Please return only the brand or company name you recognize."}
            ],
            images=[{"buffer": img_bytes}]
        )

        brand_name = response.choices[0].message.content.strip()
        return jsonify({"brandName": brand_name})

    except Exception as e:
        print("❌ OCR error:", e)
        return jsonify({"error": f"Server error: {e}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
