import os
import json
from flask import Flask, request, jsonify
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

    prompt = f"""
You are a business research assistant. Given a company name and website, return enriched lead data in JSON format.
Focus on accurate and verifiable information. Do not guess. Use real, publicly available details.

Company: {company}
Website: {website}

Return JSON with these fields only:
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

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                { "role": "system", "content": "You are a professional business research assistant." },
                { "role": "user", "content": prompt }
            ],
            temperature=0.3,
        )
        content = response.choices[0].message.content.strip()
        return jsonify(json.loads(content))

    except Exception as e:
        return jsonify({"error": str(e)}), 500
