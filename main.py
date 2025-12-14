from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List, Dict, Any, Optional, Tuple
import os
import re
import time
import pandas as pd
from openai import OpenAI
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

# =========================================================
#                     CONFIG
# =========================================================
APP_TITLE = "Nubd AI - Medical Assistant"
APP_DESC = "Arabic Medical AI Assistant API (Beta)"
APP_VERSION = "0.5.0"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

# Optional API key protection (recommended for /ask)
NUBD_API_KEY = os.getenv("NUBD_API_KEY", "").strip()

# CORS (restrict in production)
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "https://nubd-care.com,https://www.nubd-care.com,http://localhost:5173,http://localhost:3000",
).split(",")

# Basic rate limiting (in-memory, good enough for beta)
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
_rate_bucket: Dict[str, List[float]] = {}  # ip -> timestamps

# Input guardrails
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "1200"))

SAFETY_NOTICE = "تنبيه: هذه إجابة توعوية فقط وليست تشخيصاً. يجب استشارة طبيب مختص."


# =========================================================
#                     APP
# =========================================================
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESC,
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
#                 OpenAI Client
# =========================================================
client: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY not found! /ask will not work.")

# =========================================================
#                 DATASET LOADING
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

def load_datasets():
    global df_ar, df_en

    # Arabic preference order
    df_ar = _try_load_csv("medquad_ar.csv")
    if df_ar is None:
        df_ar = _try_load_csv("medquad_full_with_ar_batch.csv")
    if df_ar is None:
        df_ar = _try_load_csv("medquad_small.csv")

    # English dataset
    df_en = _try_load_csv("medquad.csv")

load_datasets()

# =========================================================
#                 HELPERS: AUTH + RATE LIMIT
# =========================================================
def get_client_ip(request: Request) -> str:
    # Render / proxies: may send x-forwarded-for
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

def rate_limit(ip: str):
    now = time.time()
    window_start = now - 60
    times = _rate_bucket.get(ip, [])
    times = [t for t in times if t >= window_start]
    if len(times) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a minute and try again.")
    times.append(now)
    _rate_bucket[ip] = times

def require_api_key(x_api_key: Optional[str]):
    # If you did not set NUBD_API_KEY in env => no auth
    if not NUBD_API_KEY:
        return
    if not x_api_key or x_api_key.strip() != NUBD_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized (missing/invalid API key).")

# =========================================================
#                 TEXT NORMALIZATION + REDACTION
# =========================================================
_AR_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
_AR_TATWEEL = "\u0640"

EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
PHONE_RE = re.compile(r"(\+?\d[\d\-\s]{7,}\d)")

def normalize_ar(text: str) -> str:
    if not text:
        return ""
    t = str(text).strip().lower()
    t = t.replace(_AR_TATWEEL, "")
    t = re.sub(_AR_DIACRITICS, "", t)
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    t = t.replace("ة", "ه")
    t = t.replace("ى", "ي")
    t = re.sub(r"\s+", " ", t)
    return t

def normalize_en(text: str) -> str:
    if not text:
        return ""
    t = str(text).strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t

def redact_pii(text: str) -> str:
    # simple redaction (do NOT store/send PII)
    if not text:
        return ""
    t = EMAIL_RE.sub("[EMAIL]", text)
    t = PHONE_RE.sub("[PHONE]", t)
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
# =========================================================
def _score_overlap(query_norm: str, row_norm: str) -> float:
    if not query_norm or not row_norm:
        return 0.0
    q_tokens = set(query_norm.split())
    r_tokens = set(row_norm.split())
    overlap = len(q_tokens.intersection(r_tokens))
    boost = 5 if query_norm in row_norm else 0
    return float(boost + overlap)

def search_ar_dataset(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    if df_ar is None or df_ar.empty:
        return []
    qn = normalize_ar(query)

    scored: List[Tuple[float, int]] = []
    for idx, row in df_ar.iterrows():
        qa = str(row.get("question_ar", "") or "")
        if not qa:
            continue
        score = _score_overlap(qn, normalize_ar(qa))
        if score > 0:
            scored.append((score, int(idx)))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, idx in scored[:top_k]:
        row = df_ar.iloc[idx]
        results.append({
            "question": row.get("question_ar", ""),
            "answer": row.get("answer_ar", ""),
            "source": row.get("source", ""),
            "focus_area": row.get("focus_area", ""),
            "row_index": int(idx),
            "score": float(score),
        })
    return results

def search_en_dataset(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    if df_en is None or df_en.empty:
        return []
    qn = normalize_en(query)

    scored: List[Tuple[float, int]] = []
    for idx, row in df_en.iterrows():
        q = str(row.get("question", "") or "")
        if not q:
            continue
        score = _score_overlap(qn, normalize_en(q))
        if score > 0:
            scored.append((score, int(idx)))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, idx in scored[:top_k]:
        row = df_en.iloc[idx]
        results.append({
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "source": row.get("source", ""),
            "focus_area": row.get("focus_area", ""),
            "row_index": int(idx),
            "score": float(score),
        })
    return results

def build_rag_context(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "لا توجد مصادر داخل قاعدة البيانات الحالية لهذا السؤال."
    lines = ["مصادر من قاعدة البيانات الداخلية (استخدمها فقط ولا تختلق معلومات):"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"\n[{i}] سؤال: {str(r.get('question','')).strip()}\n"
            f"إجابة: {str(r.get('answer','')).strip()}\n"
            f"مصدر: {str(r.get('source','')).strip()}\n"
        )
    return "\n".join(lines)

# =========================================================
#                    MODELS
# =========================================================
class SearchRequest(BaseModel):
    question: str
    top_k: int = 3

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    safety_notice: str
    sources_used: List[Dict[str, Any]] = []

class ContactRequest(BaseModel):
    name: str
    email: EmailStr
    subject: str
    message: str

# =========================================================
#                    ROOT & HEALTH
# =========================================================
@app.get("/")
def root():
    return {"message": "Nubd AI Backend is running 🚀", "beta": True, "version": APP_VERSION}

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "openai_ready": bool(client),
        "ar_rows": int(len(df_ar)) if df_ar is not None else 0,
        "en_rows": int(len(df_en)) if df_en is not None else 0,
        "version": APP_VERSION,
    }

# =========================================================
#                    SEARCH ENDPOINT
# =========================================================
@app.post("/search")
def search(req: SearchRequest, request: Request, x_api_key: Optional[str] = Header(default=None)):
    ip = get_client_ip(request)
    rate_limit(ip)
    require_api_key(x_api_key)

    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question is empty.")
    if len(q) > MAX_QUESTION_CHARS:
        raise HTTPException(status_code=400, detail="Question is too long.")

    top_k = max(1, min(int(req.top_k), 10))

    ar_results = search_ar_dataset(q, top_k=top_k)
    if ar_results:
        return {"query": q, "lang": "ar", "results": ar_results, "count": len(ar_results)}

    en_results = search_en_dataset(q, top_k=top_k)
    return {"query": q, "lang": "en", "results": en_results, "count": len(en_results)}

# =========================================================
#                    ASK ENDPOINT (RAG)
# =========================================================
@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, request: Request, x_api_key: Optional[str] = Header(default=None)):
    ip = get_client_ip(request)
    rate_limit(ip)
    require_api_key(x_api_key)

    if client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is missing on the server.")

    user_question = (req.question or "").strip()
    if not user_question:
        raise HTTPException(status_code=400, detail="Question is empty.")
    if len(user_question) > MAX_QUESTION_CHARS:
        raise HTTPException(status_code=400, detail="Question is too long.")

    # Redact PII before sending to model
    user_question_safe = redact_pii(user_question)

    # 1) Emergency gate
    if is_emergency_ar(user_question_safe):
        return AskResponse(
            answer=(
                "قد تكون الأعراض التي ذكرتها طارئة. "
                "من فضلك اتجه للطوارئ فورًا أو اتصل بالإسعاف في بلدك، "
                "ولا تنتظر ردًا عبر الإنترنت."
            ),
            safety_notice=SAFETY_NOTICE,
            sources_used=[]
        )

    # 2) Retrieve (prefer Arabic)
    results = search_ar_dataset(user_question_safe, top_k=4)
    if not results:
        results = search_en_dataset(user_question_safe, top_k=4)

    rag_context = build_rag_context(results)

    system_prompt = """
أنت "نبض" مساعد صحي عربي توعوي (ليس بديلاً عن الطبيب).
اعتمد فقط على "مصادر قاعدة البيانات" المرفقة. ممنوع اختلاق معلومات أو إضافة حقائق غير موجودة.

قواعد صارمة:
1) لا تقدّم تشخيصًا نهائيًا.
2) لا تذكر جرعات أدوية أو وصفات علاجية دقيقة.
3) لا تطلب من المستخدم بيانات شخصية (اسم/هاتف/عنوان/صور/تحاليل).
4) إذا ظهرت علامات طارئة: وجّه للطوارئ فورًا.
5) إذا لم تكفِ المصادر: قل بوضوح أنك لا تملك معلومات كافية من القاعدة الحالية.
6) استخدم أسلوب احتمالي: عدة احتمالات عامة + ما الذي يرفع/يخفض احتمالها (أسئلة توضيحية) دون جزم.

قالب الإجابة (التزم به حرفيًا):
- فهم سريع للحالة (سطرين)
- احتمالات عامة مرتبة (الأكثر شيوعًا → الأقل) دون تشخيص
- أسئلة توضيحية (3–5 أسئلة قصيرة)
- ماذا تفعل الآن؟ (خطوات عامة آمنة)
- متى تراجع طبيب؟
- متى تعتبر الحالة طارئة؟
""".strip()

    user_message = f"""
سؤال المستخدم:
{user_question_safe}

{rag_context}
""".strip()

    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
        )
        answer = completion.choices[0].message.content.strip()

        return AskResponse(
            answer=answer,
            safety_notice=SAFETY_NOTICE,
            sources_used=results
        )

    except Exception as e:
        print("OpenAI Error:", e)
        raise HTTPException(status_code=500, detail="خطأ أثناء الاتصال بنموذج الذكاء الاصطناعي.")

# =========================================================
#                    CONTACT ENDPOINT
# =========================================================
@app.post("/contact")
def send_contact_email(req: ContactRequest, request: Request):
    ip = get_client_ip(request)
    rate_limit(ip)

    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    contact_to = os.getenv("CONTACT_TO")

    if not all([smtp_host, smtp_user, smtp_pass, contact_to]):
        raise HTTPException(status_code=500, detail="SMTP settings missing on server.")

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
