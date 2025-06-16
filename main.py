from fastapi import FastAPI
from enrich import enrich_all_rows

app = FastAPI()

@app.post("/enrich")
def run_enrichment():
    updated_count = enrich_all_rows()
    return {"status": "success", "updated_rows": updated_count}

@app.get("/")
def root():
    return {"message": "Lead enrichment API is up and running."}
