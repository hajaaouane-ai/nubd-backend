import smtplib
import ssl
from email.message import EmailMessage
from pydantic import EmailStr
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import pandas as pd
from openai import OpenAI

# ============================================================
# 🚀 إعداد تطبيق FastAPI
# ============================================================
app = FastAPI(
    title="Nubd AI - Medical Assistant",
    description="Arabic Medical AI Assistant API",
    version="0.3.0",
)

# ============================================================
# 🔐 إعداد CORS (مفتوح لكل الدومينات حالياً)
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # لاحقًا يمكن قصرها على nubd-care.com
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 🔑 تهيئة عميل OpenAI
# ============================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = None

if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
    print("✅ OpenAI client initialized.")
else:
    print("⚠️ OPENAI_API_KEY not found! /ask endpoint will not work.")

# ============================================================
# 📚 تحميل الداتا (medquad_small.csv)
# ============================================================
df = None
try:
    df = pd.read_csv("medquad_small.csv", encoding="utf-8-sig")
    print(f"✅ Loaded dataset with {len(df)} rows.")
except Exception as e:
    print("⚠️ Dataset not found or failed to load:", e)

# ============================================================
# 🌐 Endpoints أساسية
# ============================================================
@app.get("/")
def root():
    return {"message": "Nubd AI Backend is running 🚀"}

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.get("/health")
def health():
    """Endpoint بسيط لاستخدامه مع Uptime مونيتور لتفادي النوم في Render."""
    return {"status": "healthy"}


# ============================================================
# 🔎 /search Endpoint – البحث في الداتا
# ============================================================
class SearchRequest(BaseModel):
    question: str
    top_k: int = 3

@app.post("/search")
def search(req: SearchRequest):
    """
    يبحث عن أسئلة مشابهة في medquad_small.csv
    ويعيد أول top_k نتائج.
    """
    if df is None:
        return {"error": "Dataset not loaded on server."}

    q = req.question.strip().lower()
    if not q:
        return {"query": req.question, "results": [], "count": 0}

    # نتأكد أن العمود موجود
    if "question_ar" not in df.columns:
        return {"error": "Column 'question_ar' not found in dataset."}

    # 🔹 بحث أسرع باستخدام pandas بدل loop كامل
    questions = df["question_ar"].fillna("").astype(str).str.lower()
    mask = questions.str.contains(q)
    matched = df[mask].head(req.top_k)

    results = []
    for idx, row in matched.iterrows():
        results.append({
            "question": str(row.get("question_ar", "")),
            "answer": str(row.get("answer_ar", "")),
            "source": str(row.get("source", "")),
            "row_index": int(idx),
        })

    return {
        "query": req.question,
        "results": results,
        "count": len(results),
    }


# ============================================================
# 🧠 /ask Endpoint – المساعد الطبي الذكي
# ============================================================
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    safety_notice: str

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """
    يأخذ سؤال طبي بالعربية، ويعيد إجابة توعوية بدون تشخيص نهائي
    مبنية على نموذج gpt-4o-mini.
    """
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is missing on the server."
        )

    user_question = req.question.strip()
    if not user_question:
        raise HTTPException(status_code=400, detail="السؤال لا يمكن أن يكون فارغاً.")

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
            model="gpt-4o-mini",   # نموذج سريع ومناسب للتوعية الطبية
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"سؤال المستخدم: {user_question}"},
            ],
            temperature=0.4,
            max_tokens=600,
        )

        # حسب نسخة مكتبة OpenAI:
        # إما completion.choices[0].message.content أو completion.choices[0].message["content"]
        choice = completion.choices[0]
        content = getattr(choice.message, "content", None)
        if content is None and isinstance(choice.message, dict):
            content = choice.message.get("content", "")

        output = (content or "").strip()

        safety_notice = (
            "تنبيه: هذه إجابة تعليمية فقط وليست تشخيصاً نهائياً، "
            "ولا تُعد خطة علاجية. يجب استشارة طبيب مختص للتأكد من أي حالة مرضية "
            "وخاصة في الحالات الطارئة أو الأعراض المقلقة."
        )

        return AskResponse(
            answer=output,
            safety_notice=safety_notice,
        )

    except Exception as e:
        print("OpenAI Error:", e)
        raise HTTPException(
            status_code=500,
            detail="حدث خطأ أثناء الاتصال بنموذج الذكاء الاصطناعي."
        )

# =============================
# Contact Form - Send Email
# =============================
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

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

    if not all([smtp_host, smtp_port, smtp_user, smtp_pass, contact_to]):
        raise HTTPException(
            status_code=500,
            detail="SMTP settings missing on the server."
        )

    try:
        # إعداد الرسالة
        msg = MIMEMultipart()
        msg["From"] = smtp_user
        msg["To"] = contact_to
        msg["Subject"] = f"رسالة جديدة من نموذج التواصل - {req.subject}"

        body = f"""
        الاسم: {req.name}
        البريد: {req.email}
        -------------------------
        الرسالة:
        {req.message}
        """

        msg.attach(MIMEText(body, "plain", "utf-8"))

        # الإرسال عبر Gmail SMTP
        server = smtplib.SMTP(smtp_host, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)
        server.quit()

        return {"status": "success", "message": "تم إرسال رسالتك بنجاح 🎉"}

    except Exception as e:
        print("Email Error:", e)
        raise HTTPException(
            status_code=500,
            detail="تعذر إرسال الرسالة. حاول مرة أخرى."
        )

# ============================================================
# 🏃 تشغيل محلي فقط (ليس في Render)
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
