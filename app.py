"""
app.py
----------------------------------
FastAPI backend for Click-Safe Fraud Detection
with Hybrid (Supervised + Unsupervised) AI Model,
Adaptive Learning, Feedback, Dashboard & Chatbot.
"""

import os
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Import Hybrid AI Model helpers
from AI_Model import (
    predict_single,
    add_feedback,
    update_model_with_feedback,
    metadata
)

app = FastAPI(title="Click-Safe Fraud Detection", version="5.0 (Hybrid + Chatbot)")

# ===============================
# CORS (Allow frontend requests)
# ===============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ⚠️ Allow all for dev; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# CONFIG
# ===============================
BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
FEEDBACK_FILE = os.path.join(BASE_DIR, "feedback.csv")

# Mount static files (images, css, js)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ===============================
# INPUT SCHEMAS
# ===============================
class EmailInput(BaseModel):
    sender_email: str
    subject: str
    content: str
    urls: str = ""
    file_types: str = ""

class UrlInput(BaseModel):
    url: str

class FeedbackInput(EmailInput):
    label: str   # "fraud" or "safe"

# ===============================
# API ENDPOINTS
# ===============================
@app.post("/predict/email")
def predict_email_api(data: EmailInput):
    """Check if an email is fraud or safe, with extra insights"""
    result = predict_single(
        sender_email=data.sender_email,
        subject=data.subject,
        content=data.content,
        urls=data.urls,
        file_types=data.file_types
    )

    # 🔹 Find similar cases from feedback.csv
    similar_cases = []
    if os.path.exists(FEEDBACK_FILE):
        df = pd.read_csv(FEEDBACK_FILE)
        if not df.empty:
            keyword = data.subject.split(" ")[0] if data.subject else ""
            similar_cases = df[
                (df["sender_email"] == data.sender_email) |
                (df["subject"].str.contains(keyword, na=False))
            ].head(3).to_dict(orient="records")

    result["similar_cases"] = similar_cases
    return result


@app.post("/predict/url")
def predict_url_api(data: UrlInput):
    """Check if a URL is fraud or safe"""
    result = predict_single(
        sender_email="",
        subject="",
        content="",
        urls=data.url,
        file_types=""
    )
    return result


@app.post("/feedback/email")
def feedback_api(data: FeedbackInput):
    """Store user feedback in feedback.csv"""
    add_feedback(
        sender_email=data.sender_email,
        subject=data.subject,
        content=data.content,
        urls=data.urls,
        file_types=data.file_types,
        label=data.label
    )
    return {"message": "✅ Feedback stored successfully."}


@app.post("/retrain")
def retrain_model():
    """Retrain/update the model with stored feedback"""
    if not os.path.exists(FEEDBACK_FILE):
        return {"message": "⚠️ No feedback available for retraining."}
    
    update_model_with_feedback()
    return {"message": "🔄 Model retrained with feedback data."}


@app.get("/history")
def get_history():
    """Return all stored feedback"""
    if os.path.exists(FEEDBACK_FILE):
        df = pd.read_csv(FEEDBACK_FILE)
        return df.to_dict(orient="records")
    return []


@app.get("/stats")
def get_stats():
    """Return fraud vs safe stats"""
    if not os.path.exists(FEEDBACK_FILE):
        return {"fraud": 0, "safe": 0, "total": 0}
    df = pd.read_csv(FEEDBACK_FILE)
    fraud_count = int((df["label"].str.lower() == "fraud").sum())
    safe_count = int((df["label"].str.lower() == "safe").sum())
    return {"fraud": fraud_count, "safe": safe_count, "total": fraud_count + safe_count}


@app.get("/model-info")
def model_info():
    """Return metadata about the currently active AI model"""
    return {
        "model_type": metadata.get("model_type", "unknown"),
        "threshold": metadata.get("final_threshold", 0.5),
        "alpha": metadata.get("alpha", 0.45),
        "train_auc": metadata.get("train_auc"),
        "fraud_class_index": metadata.get("fraud_class_index", 1),
        "train_min_score": metadata.get("train_min_score"),
        "train_max_score": metadata.get("train_max_score"),
    }

# ===============================
# CHATBOT ENDPOINT
# ===============================
@app.post("/chat")
async def chatbot_api(request: Request):
    """Simple rule-based chatbot for demo"""
    data = await request.json()
    query = data.get("query", "").lower()

    if "help" in query:
        response = "You can check fraud in emails, fraud links, and get risk stats on the dashboard."
    elif "dashboard" in query:
        response = "The dashboard shows fraud vs safe stats and retraining options."
    elif "features" in query:
        response = "Click-Safe features include Email Checker, Fraud Links, Chatbot, and AI-powered monitoring."
    elif "email" in query:
        response = "Go to the Email Checker tool to scan suspicious emails."
    elif "link" in query or "url" in query:
        response = "Use the Fraud Links tool to detect phishing or unsafe websites."
    else:
        response = "I'm your assistant 🤖. Try asking about 'help', 'dashboard', 'features', 'email', or 'links'."

    return {"response": response}

# ===============================
# FRONTEND ROUTES
# ===============================
@app.get("/")
def home_page():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

@app.get("/email")
def email_page():
    return FileResponse(os.path.join(BASE_DIR, "email.html"))

@app.get("/url")
def url_page():
    return FileResponse(os.path.join(BASE_DIR, "url.html"))

@app.get("/chatbot")
def chatbot_page():
    return FileResponse(os.path.join(BASE_DIR, "chatbot.html"))

