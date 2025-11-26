from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

app = FastAPI(
    title="Nubd AI - Medical Assistant",
    description="Arabic Medical AI Assistant API",
    version="0.1.0",
)

# -----------------------------
# Load medical data (placeholder)
# -----------------------------
try:
    df = pd.read_csv("medquad_full_with_ar.csv", encoding="utf-8-sig")
    print(f"Loaded dataset with {len(df)} rows.")
except:
    df = None
    print("⚠️ Dataset not found. Upload medquad_full_with_ar.csv to Render later.")


# -----------------------------
# Root endpoint
# -----------------------------
@app.get("/")
def root():
    return {"message": "Nubd AI Backend is running 🚀"}


# -----------------------------
# Health check
# -----------------------------
@app.get("/ping")
def ping():
    return {"status": "ok"}


# -----------------------------
# Search request model
# -----------------------------
class SearchRequest(BaseModel):
    question: str
    top_k: int = 3


# -----------------------------
# Simple Arabic search endpoint
# -----------------------------
@app.post("/search")
def search(req: SearchRequest):

    if df is None:
        return {"error": "Dataset not loaded on server."}

    q = req.question.strip().lower()

    results = []

    for idx, row in df.iterrows():
        question_ar = str(row.get("question_ar", "")).strip().lower()
        answer_ar = str(row.get("answer_ar", "")).strip()

        if q in question_ar:
            results.append({
                "question": row.get("question_ar", ""),
                "answer": row.get("answer_ar", ""),
                "source": row.get("source", ""),
                "row_index": int(idx)
            })

        if len(results) >= req.top_k:
            break

    return {
        "query": req.question,
        "results": results,
        "count": len(results)
    }


# -----------------------------
# Run locally (Render uses this)
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
