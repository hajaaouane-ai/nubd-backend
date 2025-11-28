from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import pandas as pd
from openai import OpenAI

app = FastAPI(
    title="Nubd AI - Medical Assistant",
    description="Arabic Medical AI Assistant API",
    version="0.2.0",
)

# -----------------------------
# CORS (FIXED)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# OpenAI Client
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    print("⚠️ OPENAI_API_KEY not found! /ask will not work.")

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
# Basic Endpoints
# -----------------------------
@app.get("/")
def root():
    return {"message": "Nubd AI Backend is running 🚀"}

@app.get("/ping")
def ping():
    return {"status": "ok"}

# -----------------------------
# SEARCH Endpoint
# -----------------------------
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
# ASK Endpoint (AI Medical Assistant)
# -----------------------------
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    safety_notice: str

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):

    if client is None:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is missing on the server."
        )

    user_question = req.question.strip()
    if not user_question:
        raise HTTPException(400, "السؤال لا يمكن أن يكون فارغاً.")

    system_prompt = """
أنت مساعد طبي عربي ذكي يستخدم تحليل احتمالات مستوحى من الفيزياء الكمية (Quantum-inspired reasoning).
تحدّث بالعربية الواضحة والمبسّطة، واتّبع القواعد التالية:

1. لا تعطي تشخيص نهائي، فقط احتمالات عامة.
2. لا تصف أدوية بجرعات محددة.
3. إن كان السؤال يشير لحالة طارئة: ألم صدر حاد، ضيق نفس شديد، أعراض جلطة، نزيف حاد → اطلب الذهاب للطوارئ فوراً.
4. استخدم نمط الإجابة التالي:
   - شرح عام للسؤال
   - أكثر الأسباب المحتملة (بأسلوب احتمالات مثل superposition)
   - متى يجب زيارة الطبيب
   - متى يجب التوجه للطوارئ
"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"سؤال المستخدم: {user_question}"}
            ],
            temperature=0.4
        )

        output = completion.choices[0].message.content.strip()

        safety_notice = (
            "تنبيه: هذه إجابة تعليمية فقط وليست تشخيصاً نهائياً. "
            "يجب استشارة طبيب مختص للتأكد من أي حالة مرضية."
        )

        return AskResponse(
            answer=output,
            safety_notice=safety_notice
        )

    except Exception as e:
        print("OpenAI Error:", e)
        raise HTTPException(
            status_code=500,
            detail="حدث خطأ أثناء الاتصال بنموذج الذكاء الاصطناعي."
        )

# -----------------------------
# Local Run
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
