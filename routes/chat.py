"""
routes/chat.py
Dermatology-only chatbot powered by Groq Llama-3.

POST /api/chat — Send a question, get a skin-condition answer (JWT required)

The system prompt strictly limits responses to 4 skin conditions.
Any off-topic question gets a clear refusal message.
"""
import os
import requests
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from utils.auth import get_current_user_id
from utils.logger import get_logger

logger = get_logger("routes.chat")
router = APIRouter(prefix="/api", tags=["Chat"])

GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

# System prompt locks the bot to dermatology only
SYSTEM_PROMPT = """You are DermaBot, an AI assistant for Derma Vision — a skin cancer screening app.

You ONLY answer questions about these 4 skin conditions:
1. Melanoma (MEL) — most dangerous skin cancer, high mortality
2. Melanocytic Nevi (NV) — common moles, usually benign
3. Basal Cell Carcinoma (BCC) — most common skin cancer
4. Actinic Keratosis (AKIEC) — precancerous condition

You CAN:
- Explain what each condition is and its symptoms
- Explain what the AI diagnosis result means
- Give general sun protection and skincare advice
- Explain confidence scores and why results may be inconclusive

You MUST NOT:
- Diagnose, prescribe medication, or give treatment plans
- Answer questions outside dermatology — if asked, respond exactly with:
  "I can only assist with skin-related questions about Melanoma, Melanocytic Nevi, Basal Cell Carcinoma, and Actinic Keratosis."
- Give specific drug dosages

Always end every response with:
"⚠️ This AI tool does not replace professional medical diagnosis. Please consult a dermatologist."
"""


class ChatRequest(BaseModel):
    message:   str
    diagnosis: str = ""   # optional — last scan result for context


@router.post("/chat")
def chat(
    body: ChatRequest,
    user_id: int = Depends(get_current_user_id)
):
    if not body.message.strip():
        raise HTTPException(400, "Message cannot be empty")

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        logger.error("GROQ_API_KEY not set in .env")
        raise HTTPException(503, "Groq API key not configured on server")

    # Prepend scan result to the user message if provided
    user_text = body.message
    if body.diagnosis:
        user_text = f"My AI scan result was: {body.diagnosis}\n\nMy question: {body.message}"

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_text},
        ],
        "max_tokens":  512,
        "temperature": 0.4,
    }

    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
            },
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        reply = resp.json()["choices"][0]["message"]["content"]
        logger.info(f"Chat response sent to user_id={user_id}")

    except requests.exceptions.Timeout:
        logger.error("Groq API timed out")
        raise HTTPException(504, "Groq API timed out. Please try again.")
    except requests.exceptions.HTTPError as e:
        logger.error(f"Groq API HTTP error: {e}")
        raise HTTPException(502, f"Groq API error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected chat error: {e}")
        raise HTTPException(500, f"Unexpected error: {str(e)}")

    return {"reply": reply}