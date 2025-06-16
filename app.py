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
        print("❌ Backend error:", e)
        return jsonify({"error": f"Server error: {e}"}), 500
