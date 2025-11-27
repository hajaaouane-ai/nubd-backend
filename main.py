from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

app = FastAPI(
    title="Nubd AI - Medical Assistant",
    description="Arabic Medical AI Assistant API",
    version="0.1.0",
)

# -----------------------------
# FIXED CORS (FULL)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # ← لمنع مشاكل CORS نهائياً الآن
    allow_credentials=True,
    allow_methods=["*"],          # ← يسمح بكل الطرق (GET, POST, OPTIONS)
    allow_headers=["*"],          # ← يسمح بكل الهيدرز
)

# -----------------------------
# Load dataset
# -----------------------------
try:
    df = pd.read_csv("medquad_small.csv", encoding="utf-8-sig")
    print(f"Loaded dataset with {len(df)} rows.")
except Exception as e:
    df = None
    print("⚠️ Dataset not found:", e)

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
def root():
    return {"message": "Nubd AI Backend is running 🚀"}

@app.get("/ping")
def ping():
    return {"status": "ok"}

class SearchRequest(BaseModel):
    question: str
    top_k: int = 3

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
# Run locally
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
