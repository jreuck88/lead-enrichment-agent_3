from flask import Flask, request, jsonify
import openai
import os

app = Flask(__name__)

openai.api_key = os.environ.get("OPENAI_API_KEY")

def enrich_row(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0.2
        )
        content = response['choices'][0]['message']['content']
        return eval(content)  # <- if you're returning dicts like "{'key': 'value'}"
    except Exception as e:
        return {"error": str(e)}

@app.route("/")
def home():
    return "🟢 CRM Lead Enrichment Agent is live."

@app.route("/enrich", methods=["POST"])
def enrich():
    try:
        data = request.get_json()
        company = data.get("company_name", "")
        website = data.get("website", "")

        prompt = f"""Enrich this company:
Company: {company}
Website: {website}
Return JSON with the following keys only:
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

        return jsonify(enriched)

    except Exception as e:
        return jsonify({"error": f"Server error: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
