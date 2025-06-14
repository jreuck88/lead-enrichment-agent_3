import os, json
from flask import Flask, request, jsonify
import openai

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")  # set this in Render secrets

@app.route("/")
def home():
    return "✅ Lead Enrichment & OCR Agent is live."

@app.route("/enrich", methods=["POST"])
def enrich():
    data = request.get_json(force=True)
    company = data.get("company_name")
    website = data.get("website")
    if not company or not website:
        return jsonify({"error": "Missing company_name or website"}), 400

    prompt = f"""
You are a business research assistant. Given a company name and website, return enriched lead data in JSON:
Company: {company}
Website: {website}
Return ONLY this JSON object with keys:
CompanyName, CompanyEmail, Location, BestPOC, POCEmail,
POCLinkedIn, InstagramURL, LinkedInURL, Website,
CompanyServices, ValueProp, CompanySize, AnnualRevenue, LeadScore
"""
    resp = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role":"system","content":"You are a professional research assistant."},
            {"role":"user","content":prompt}
        ],
        temperature=0.3
    )
    content = resp.choices[0].message.content.strip()
    # strip markdown fences
    content = content.replace("```json","").replace("```","").strip()
    return jsonify(json.loads(content))


@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    data = request.get_json(force=True)
    b64  = data.get("image_base64")
    if not b64:
        return jsonify({"error":"Missing image_base64"}), 400

    prompt = f"""
You are an OCR assistant. Given the following base64-encoded image, extract:
1) the brand or company name
2) its official website

Return ONLY a JSON with keys "brandName" and "brandWebsite".

{b64}
"""
    resp = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role":"system","content":"You are an expert OCR assistant."},
            {"role":"user","content":prompt}
        ],
        temperature=0.0
    )
    content = resp.choices[0].message.content.strip()
    content = content.replace("```","").strip()
    return jsonify(json.loads(content))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
