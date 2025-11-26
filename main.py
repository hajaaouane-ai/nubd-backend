from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI(
    title="Nubd AI - Medical Assistant",
    description="Arabic Medical AI Assistant API",
    version="0.1.0",
)

# -----------------------------
# CORS settings
# -----------------------------
origins = [
    "https://nubd-care.com",
    "http://localhost:5173",  # للتطوير المحلي لاحقًا
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load medical dataset
# -----------------------------
try:
    df = pd.read_csv("medquad_small.csv", encoding="utf-8-sig")
    print(f"Loaded dataset with {len(df)} rows.")
except Exception as e:
    df = None
    print("⚠️ Dataset not found or failed to load:", e)


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
# Simple Arabic keyword search
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

        if q in question_ar:  # بحث بالكلمات
            results.append({
                "question": row.get("question_ar", ""),
                "answer": row.get("answer_ar", ""),
                "source": row.get("source", ""),
                "row_index": int(idx)
            })

        if len(results) >= req.top
