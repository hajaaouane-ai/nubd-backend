from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import pandas as pd
from openai import OpenAI
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

app = FastAPI(
    title="Nubd AI - Medical Assistant",
    description="Arabic Medical AI Assistant API",
    version="0.3.0",
)

# ==============================
#        CORS Settings
# ==============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
#       OpenAI Client
# ==============================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = None

if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    print("⚠️ OPENAI_API_KEY not found! /ask will not work.")


# ==============================
#     Load Dataset (Search)
# ==============================
try:
    df = pd.read_csv("medquad_small.csv", encoding="utf-8-sig")
    print(f"Loaded dataset with {len(df)} rows.")
except Exception as e:
    df = None
    print("⚠️ Dataset not found:", e)


# ==============================
#      Root & Ping
# ==============================
@app.get("/")
def root():
    return {"message": "Nubd AI Backend is running 🚀"}

@app.get("/ping")
def ping():
    return {"status": "ok"}


# ==============================
#      SEARCH Endpoint
# ==============================
class SearchRequest(BaseModel):
    question: str
    top_k: int = 3

@app.post("/search")
def search(req: SearchRequest):

    if df is None:
        return {"error": "Dataset not loaded on server."}

    query = req.question.strip().lower()
    results = []

    for idx, row in df.iterrows():
        question_ar = str(row.get("question_ar", "")).strip().lower()

        if query in question_ar:
            results.append({
                "question": row.get("question_ar", ""),
                "answer": row.get("answer_ar", ""),
                "source": row.get("source", ""),
                "row_index": int(idx)
            })

        if len(results) >= req.top_k:
            break

    return {"query": req.question, "results": results, "count": len(results)}


# ==============================
#        ASK Endpoint
# ==============================
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

    system_prompt = """
أنت مساعد طبي عربي ذكي يستخدم تحليل احتمالات مستوحى من الفيزياء الكمية (Quantum-inspired reasoning).
اتبع القواعد التالية:
1. لا تقدّم تشخيصًا نهائيًا.
2. لا تصف جرعات أدوية.
3. إن كانت هناك أعراض طارئة → اطلب الذهاب للطوارئ.
4. استخدم لغة عربية بسيطة.
"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ],
            temperature=0.4
        )

        answer = completion.choices[0].message.content.strip()

        return AskResponse(
            answer=answer,
            safety_notice="تنبيه: هذه إجابة توعوية فقط وليست تشخيصاً. يجب استشارة طبيب مختص."
        )

    except Exception as e:
        print("OpenAI Error:", e)
        raise HTTPException(status_code=500, detail="خطأ أثناء الاتصال بنموذج الذكاء الاصطناعي.")


# ==============================
#       CONTACT Endpoint
# ==============================
class ContactRequest(BaseModel):
    name: str
    email: str
    subject: str
    message: str

@app.post("/contact")
def send_contact_email(req: ContactRequest):

    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    contact_to = os.getenv("CONTACT_TO")

    if not all([smtp_host, smtp_user, smtp_pass, contact_to]):
        raise HTTPException(
            status_code=500,
            detail="SMTP settings missing on server."
        )

    try:
        msg = MIMEMultipart()
        msg["From"] = smtp_user
        msg["To"] = contact_to
        msg["Subject"] = f"رسالة جديدة من نموذج التواصل - {req.subject}"

        body = f"""
        الاسم: {req.name}
        البريد: {req.email}
        الموضوع: {req.subject}
        -------------------------
        الرسالة:
        {req.message}
        """

        msg.attach(MIMEText(body, "plain", "utf-8"))

        server = smtplib.SMTP(smtp_host, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)
        server.quit()

        return {"status": "success", "message": "تم إرسال رسالتك بنجاح 🎉"}

    except Exception as e:
        print("Email Error:", e)
        raise HTTPException(status_code=500, detail="تعذر إرسال الرسالة حالياً.")


# ==============================
#          RUN SERVER
# ==============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
