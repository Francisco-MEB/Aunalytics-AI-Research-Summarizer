import os
import json
from uuid import uuid4
from fastapi import APIRouter, UploadFile, File, Form

import google.generativeai as genai

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_KEY)

MODEL_NAME = "gemini-2.0-flash" 
model = genai.GenerativeModel(MODEL_NAME)

router = APIRouter(tags=["Chat"])

@router.post("/")
async def chat_endpoint(
    message: str = Form(...),
    history: str = Form("[]"),
    file: UploadFile = File(None),
):
    """
    Pure LLM chat. Explain as if you are talking to a person from a non-technical background.
    Files are optional—they are read and included directly into the prompt.
    """

    # ------------------------------
    # 1. Handle file upload (optional)
    # ------------------------------
    file_context = ""
    if file:
        try:
            content = (await file.read()).decode("utf-8", errors="ignore")
            file_context = f"\n\nFILE CONTENT:\n{content}\n\n"
        except:
            file_context = "\n\n[File could not be read.]\n\n"

    # ------------------------------
    # 2. Parse chat history
    # ------------------------------
    try:
        history_items = json.loads(history)
    except:
        history_items = []

    # Convert frontend history → Gemini format
    gemini_messages = []
    for item in history_items:
        role = item.get("role", "user")
        text = item.get("content", "")
        gemini_messages.append({"role": role, "parts": [text]})

    # ------------------------------
    # 3. Add user message
    # ------------------------------
    full_input = (
        "You are a helpful academic assistant.\n"
        "Answer normally. Do NOT require external context. Do NOT use section headers. Do NOT use bullet points. Do NOT use bold or markdown or italics; just use plain text. Do NOT list information — instead blend it into prose.\n"
        "If a file was uploaded, use its content.\n\n"
        f"{file_context}"
        f"USER QUESTION:\n{message}\n"
    )

    gemini_messages.append({"role": "user", "parts": [full_input]})
 
    response = model.generate_content(gemini_messages)
    assistant_reply = response.text

    return {
        "response": assistant_reply,
        "file_used": bool(file),
        "model": MODEL_NAME
    }
