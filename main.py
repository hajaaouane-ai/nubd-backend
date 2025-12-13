from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List, Dict, Any, Optional
import os
import re
import pandas as pd
from openai import OpenAI
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

# =========================================================
#                     APP
# =========================================================
app = FastAPI(
    title="Nubd AI - Medical Assistant",
    description="Arabic Medical AI Assistant API (Beta)",
    version="0.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Beta فقط. لاحقًا حدده لدومينك.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
#                 OpenAI Client
# =========================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY not found! /ask will not work.")

# =========================================================
#                 DATASET LOADING
#   Preference order:
#   1) medquad_ar.csv (if you create it later)
#   2) medquad_full_with_ar_batch.csv (your 60 Arabic Q/A)
#   3) medquad_small.csv (legacy)
#   4) medquad.csv (English full dataset 16k)
# =========================================================
df_ar: Optional[pd.DataFrame] = None
df_en: Optional[pd.DataFrame] = None

def _try_load_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        if os.path.exists(path):
            d = pd.read_csv(path, encoding="utf-8-sig")
            print(f"✅ Loaded {path} with {len(d)} rows.")
            return d
    except Exception as e:
        print(f"⚠️ Failed loading {path}: {e}")
    return None

# Arabic dataset candidates
df_ar = _try_load_csv("medquad_ar.csv")
if df_ar is None:
    df_ar = _try_load_csv("medquad_full_with_ar_batch.csv")
if df_ar is None:
    df_ar = _try_load_csv("medquad_small.csv")

# English dataset
df_en = _try_load_csv("medquad.csv")

# =========================================================
#                 TEXT NORMALIZATION
# =========================================================
_AR_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
_AR_TATWEEL = "\u0640"

def normalize_ar(text: str) -> str:
    if not text:
        return ""
    t = str(text).strip().lower()
    t = t.replace(_AR_TATWEEL, "")
    t = re.sub(_AR_DIACRITICS, "", t)
    # توحيد شائع
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    t = t.replace("ة", "ه")
    t = t.replace("ى", "ي")
    # تنظيف رموز بسيطة
    t = re.sub(r"\s+", " ", t)
    return t

def normalize_en(text: str) -> str:
    if not text:
        return ""
    t = str(text).strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t

# =========================================================
#                 EMERGENCY FILTER
# =========================================================
def is_emergency_ar(text: str) -> bool:
    t = normalize_ar(text)
    emergency_keywords = [
        "الم شديد في الصدر", "الم الصدر", "ضيق تنفس", "اختناق",
        "اغماء", "فقدان الوعي", "نزيف شديد", "نزيف قوي",
        "ضعف مفاجئ", "شلل", "تلعثم", "صعوبه كلام",
        "صداع مفاجئ شديد", "الم شديد في الراس",
        "تشنجات", "زرقة", "افكار انتحار", "ايذاء النفس"
    ]
    return any(k in t for k in emergency_keywords)

# =========================================================
#                 SEARCH (RAG RETRIEVAL)
#   - Lightweight scoring (substring + token overlap)
# =========================================================
def search_ar_dataset(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    if df_ar is None or df_ar.empty:
        return []

    qn = normalize_ar(query)
    q_tokens = set(qn.split())

    scored = []
    for idx, row in df_ar.iterrows():
        qa = str(row.get("question_ar", "") or "")
        aa = str(row.get("answer_ar", "") or "")
        if not qa:
            continue

        qrow = normalize_ar(qa)
        score = 0

        # substring boost
        if qn and qn in qrow:
            score += 5

        # token overlap
        row_tokens = set(qrow.split())
        overlap = len(q_tokens.intersection(row_tokens))
        score += overlap

        if score > 0:
            scored.append((score, idx, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, idx, row in scored[:top_k]:
        results.append({
            "question": row.get("question_ar", ""),
            "answer": row.get("answer_ar", ""),
            "source": row.get("source", ""),
            "focus_area": row.get("focus_area", ""),
            "row_index": int(idx),
            "score": float(score)
        })
    return results

def search_en_dataset(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    if df_en is None or df_en.empty:
        return []

    qn = normalize_en(query)
    q_tokens = set(qn.split())

    scored = []
    for idx, row in df_en.iterrows():
        q = str(row.get("question", "") or "")
        a = str(row.get("answer", "") or "")
        if not q:
            continue

        qrow = normalize_en(q)
        score = 0

        if qn and qn in qrow:
            score += 5

        row_tokens = set(qrow.split())
        overlap = len(q_tokens.intersection(row_tokens))
        score += overlap

        if score > 0:
            scored.append((score, idx, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, idx, row in scored[:top_k]:
        results.append({
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "source": row.get("source", ""),
            "focus_area": row.get("focus_area", ""),
            "row_index": int(idx),
            "score": float(score)
        })
    return results

def build_rag_context(results: List[Dict[str, Any]], lang: str) -> str:
    if not results:
        return "لا توجد مصادر داخل قاعدة البيانات الحالية لهذا السؤال."

    lines = ["مصادر من قاعدة البيانات الداخلية (للإجابة دون اختراع معلومات):"]
    for i, r in enumerate(results, 1):
        q = (r.get("question") or "").strip()
        a = (r.get("answer") or "").strip()
        s = (r.get("source") or "").strip()
        fa = (r.get("focus_area") or "").strip()
        ri = r.get("row_index")
        lines.append(
            f"\n[{i}]"
            f"\nQuestion: {q}"
            f"\nAnswer: {a}"
            f"\nSource: {s}"
            f"\nFocus: {fa}"
            f"\nRow: {ri}"
        )
    return "\n".join(lines)

# =========================================================
#                    ROOT & PING
# =========================================================
@app.get("/")
def root():
    return {"message": "Nubd AI Backend is running 🚀", "beta": True}

@app.get("/ping")
def ping():
    return {"status": "ok"}

# =========================================================
#                    SEARCH ENDPOINT
# =========================================================
class SearchRequest(BaseModel):
    question: str
    top_k: int = 3

@app.post("/search")
def search(req: SearchRequest):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question is empty.")

    top_k = max(1, min(int(req.top_k), 10))

    # Prefer Arabic if available
    ar_results = search_ar_dataset(q, top_k=top_k)
    if ar_results:
        return {"query": q, "lang": "ar", "results": ar_results, "count": len(ar_results)}

    en_results = search_en_dataset(q, top_k=top_k)
    return {"query": q, "lang": "en", "results": en_results, "count": len(en_results)}

# =========================================================
#                    ASK ENDPOINT (RAG)
# =========================================================
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    safety_notice: str
    sources_used: List[Dict[str, Any]] = []

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    if client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is missing on the server.")

    user_question = (req.question or "").strip()
    if not user_question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    # 1) Emergency gate
    if is_emergency_ar(user_question):
        return AskResponse(
            answer=(
                "قد تكون الأعراض التي ذكرتها طارئة. "
                "من فضلك اتجه للطوارئ فورًا أو اتصل بالإسعاف في بلدك، "
                "ولا تنتظر ردًا عبر الإنترنت."
            ),
            safety_notice="تنبيه: هذه إجابة توعوية فقط وليست تشخيصاً. يجب استشارة طبيب مختص.",
            sources_used=[]
        )

    # 2) Retrieve (prefer Arabic)
    results = search_ar_dataset(user_question, top_k=4)
    lang = "ar"
    if not results:
        results = search_en_dataset(user_question, top_k=4)
        lang = "en"

    rag_context = build_rag_context(results, lang=lang)

    system_prompt = """
أنت "نبض" مساعد صحي عربي توعوي (ليس بديلاً عن الطبيب).
مهمتك: شرح الأعراض بهدوء وبأسلوب بسيط، اعتمادًا على المصادر المرفقة، دون اختراع معلومات.

قواعد صارمة:
1) لا تقدّم تشخيصًا نهائيًا.
2) لا تذكر جرعات أدوية أو وصفات علاجية دقيقة.
3) لا تطلب من المستخدم صور/تقارير/بيانات شخصية.
4) إذا ظهرت علامات طارئة: وجّه للطوارئ فورًا.
5) إذا لم تكفِ المصادر: قل بوضوح أنك لا تملك معلومات كافية من القاعدة الحالية.
6) استخدم أسلوب احتمالي (Quantum-inspired): عدة احتمالات عامة + ماذا يغيّر الاحتمالات (أسئلة توضيحية) دون جزم.

قالب الإجابة (التزم به):
- فهم سريع للحالة (سطرين)
- احتمالات عامة مرتبة (الأكثر شيوعًا → الأقل) دون تشخيص
- أسئلة توضيحية (3–5 أسئلة قصيرة)
- ماذا تفعل الآن؟ (خطوات عامة آمنة)
- متى تراجع طبيب؟
- متى تعتبر الحالة طارئة؟
"""

    user_message = f"""
سؤال المستخدم:
{user_question}

{rag_context}
"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_message.strip()},
            ],
            temperature=0.3
        )

        answer = completion.choices[0].message.content.strip()

        # remove internal English labels if any (light cleanup)
        answer = answer.replace("Question:", "").replace("Answer:", "").strip()

        return AskResponse(
            answer=answer,
            safety_notice="تنبيه: هذه إجابة توعوية فقط وليست تشخيصاً. يجب استشارة طبيب مختص.",
            sources_used=results
        )

    except Exception as e:
        print("OpenAI Error:", e)
        raise HTTPException(status_code=500, detail="خطأ أثناء الاتصال بنموذج الذكاء الاصطناعي.")

# =========================================================
#                    CONTACT ENDPOINT
# =========================================================
class ContactRequest(BaseModel):
    name: str
    email: EmailStr
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
        raise HTTPException(status_code=500, detail="SMTP settings missing on server.")

    # Safety: keep contact for feedback/tech only (not medical consultation)
    header_note = (
        "ملاحظة: نموذج التواصل مخصص للملاحظات التقنية/الاقتراحات فقط، "
        "ولا يمكن تقديم استشارات طبية عبر البريد."
    )

    try:
        msg = MIMEMultipart()
        msg["From"] = smtp_user
        msg["To"] = contact_to
        msg["Subject"] = f"Nubd Contact - {req.subject}"

        body = f"""{header_note}

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

# =========================================================
#                      RUN SERVER
# =========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
