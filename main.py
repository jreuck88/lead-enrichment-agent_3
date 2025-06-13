import os
import json
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)

# Initialize OpenAI client using Render environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/")
def home():
    return "✅ Lead Enrichment Agent is live."

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
                { "role": "system", "content": "You are a professional research assistant." },
                { "role": "user", "content": prompt }
            ],
            temperature=0.3
        )

        content = response.choices[0].message.content.strip()

        # Extract JSON if wrapped in markdown
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()

        # Attempt to parse and return JSON
        parsed = json.loads(content)
        return jsonify(parsed)

    except Exception as e:
        print("❌ Backend error:", str(e))
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
