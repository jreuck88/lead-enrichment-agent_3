import os
import json
from flask import Flask, request, jsonify
import gspread
from openai import OpenAI
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Setup Google Sheets
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
CREDS_FILE = "google-creds.json"
SPREADSHEET_ID = "1thZnhvqC_rZZH4Ixa7a2PoXnuq8gnWXBJGxsqjUk3KU"
SHEET_NAME = "Agent"

def get_sheet():
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE, SCOPE)
    gc = gspread.authorize(creds)
    return gc.open_by_key(SPREADSHEET_ID).worksheet(SHEET_NAME)

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
        print("❌ Backend error:", e)
        return {"error": f"{e}"}

@app.route("/")
def home():
    return jsonify({"message": "Lead Enrichment API is live."})

@app.route("/enrich", methods=["POST"])
def enrich():
    try:
        sheet = get_sheet()
        records = sheet.get_all_records(head=9)
        headers = sheet.row_values(9)

        updated_count = 0

        for i, row in enumerate(records):
            row_num = i + 10
            if row.get("Enriched") == "1":
                continue

            prompt = f"Enrich this company: {row.get('Company Name', '')}, located in {row.get('Location', '')}. Provide JSON format for Company Services, Value Proposition, Best POC (Name, Role, Email), Company Size, Revenue, LinkedIn, Instagram, Website."

            enriched = enrich_row(prompt)
            if "error" in enriched:
                continue

            for key, val in enriched.items():
                if key in headers:
                    col = headers.index(key) + 1
                    sheet.update_cell(row_num, col, val)

            # Set "Enriched" = 1
            sheet.update_cell(row_num, headers.index("Enriched") + 1, "1")
            sheet.update_cell(row_num, headers.index("Date Added") + 1, datetime.now().strftime("%Y-%m-%d"))
            updated_count += 1

        return jsonify({"status": "done", "updated_rows": updated_count})

    except Exception as e:
        print("❌ Backend error:", e)
        return jsonify({"error": f"Server error: {e}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
