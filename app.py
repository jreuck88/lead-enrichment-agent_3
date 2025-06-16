import os
import json
from flask import Flask, request, jsonify
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
from openai import OpenAI

# 🔧 Flask app setup
app = Flask(__name__)

# 🔐 OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 📊 Google Sheets config
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
CREDS_FILE = "/etc/secrets/service_account.json"
SPREADSHEET_ID = "1thZnhvqC_rZZH4Ixa7a2PoXnuq8gnWXBJGxsqjUk3KU"
SHEET_NAME = "Agent"
HEADER_ROW_INDEX = 1  # Row 1 = top row

def get_sheet():
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, SCOPE)
    gc = gspread.authorize(creds)
    return gc.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)

def log_progress(sheet, message):
    try:
        sheet.update("B2", [[datetime.now().strftime("%Y-%m-%d %H:%M:%S")]])
    except Exception as e:
        print("⚠️ Logging failed:", e)

def enrich_row(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional research assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```json") or content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        print("❌ OpenAI error:", e)
        return {"error": str(e)}

@app.route("/")
def home():
    return jsonify({"message": "✅ CRM Lead Enrichment API is live."})

@app.route("/enrich", methods=["POST"])
def enrich():
    try:
        data = request.get_json()
        company = data.get("company_name", "")
        website = data.get("website", "")

        prompt = f"""Enrich this company:
Company: {company}
Website: {website}

Return JSON with only these fields:
- CompanyName
- CompanyEmail
- Location
- BestPOC
- POCEmail
- POCLinkedIn
- InstagramURL
- LinkedInURL
- Website
- CompanyServices
- ValueProp
- CompanySize
- AnnualRevenue
- LeadScore
"""

        enriched = enrich_row(prompt)
        if "error" in enriched:
            return jsonify({"error": enriched["error"]})

        # Optional logging
        try:
            sheet = get_sheet()
            log_progress(sheet, f"Enriched: {company}")
        except Exception as log_err:
            print("⚠️ Could not log enrichment:", log_err)

        return jsonify(enriched)

    except Exception as e:
        print("❌ Server error:", e)
        return jsonify({"error": f"Server error: {e}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
