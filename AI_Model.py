"""
AI_Model.py
----------------------------------
Strict Hybrid Fraud Email/URL Detection
- TF-IDF + IsolationForest (unsupervised anomaly detection)
- Strong rule-based overrides (keywords, URLs, file types)
- Adaptive retraining with feedback.csv
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest

# ===============================
# CONFIG
# ===============================
base_path = os.path.dirname(os.path.abspath(__file__))

out_model_path = os.path.join(base_path, "fraud_detection_model.pkl")
out_vectorizer_path = os.path.join(base_path, "fraud_vectorizer.pkl")
out_meta_path = os.path.join(base_path, "fraud_metadata.pkl")
processed_cases_path = os.path.join(base_path, "fraud_cases_processed.csv")
feedback_path = os.path.join(base_path, "feedback.csv")

# ===============================
# HELPERS
# ===============================
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+|https\S+", " url ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return text.strip()

def normalize_anomaly(score: float, train_min: float, train_max: float) -> float:
    if train_max == train_min:
        return 0.0
    norm = (train_max - score) / (train_max - train_min)
    return float(max(0.0, min(1.0, norm)))

def compute_rule_score_and_reasons(text: str, file_types: str, urls: str, meta: dict):
    reasons = []
    lowtext = text.lower()
    url_text = str(urls).lower()

    score = 0.0

    # 🚨 Hard fraud keywords
    fraud_keywords = [
        "urgent", "lottery", "prince", "inheritance", "transfer",
        "bank account", "verify", "password", "login", "ssn"
    ]
    for kw in fraud_keywords:
        if kw in lowtext:
            score += 0.4
            reasons.append(f"Keyword triggered: {kw}")

    # 🚨 Dangerous file types
    for f in meta.get("dangerous_files", [".exe", ".scr", ".zip", ".rar"]):
        if f in str(file_types).lower():
            score += 0.5
            reasons.append(f"Dangerous file attached: {f}")
            break

    # 🚨 Dangerous URLs
    if "http" in url_text or "www." in url_text:
        score += 0.2
        reasons.append("Contains URL")
    for d in meta.get("dangerous_urls", ["bit.ly", "tinyurl", "phish", "malware"]):
        if d in url_text:
            score += 0.5
            reasons.append(f"Suspicious URL pattern: {d}")
            break

    score = min(1.0, score)
    return score, reasons

def detect_case_type(text: str, file_types: str, urls: str):
    low = text.lower()
    if any(k in low for k in ["prince", "inheritance", "transfer"]):
        return "Nigerian Prince / Inheritance Scam"
    if any(k in low for k in ["lottery", "prize", "winner"]):
        return "Lottery / Prize Scam"
    if any(k in low for k in ["password", "verify", "bank", "login", "account"]):
        return "Credential Theft / Phishing"
    if any(f in file_types.lower() for f in [".exe", ".scr", ".zip", ".rar"]):
        return "Malicious Attachment / Malware"
    return "Unknown / General Scam"

# ===============================
# LOAD TRAINED ARTIFACTS
# ===============================
if os.path.exists(out_model_path) and os.path.exists(out_vectorizer_path) and os.path.exists(out_meta_path):
    model = joblib.load(out_model_path)
    vectorizer = joblib.load(out_vectorizer_path)
    metadata = joblib.load(out_meta_path)
else:
    raise SystemExit("❌ Model artifacts not found. Run training first.")

# ===============================
# PREDICTION WRAPPER
# ===============================
def predict_single(sender_email, subject, content, urls, file_types):
    """Strict fraud prediction with hard rules + ML anomaly"""

    combined = f"{sender_email} {subject} {content} {urls} {file_types}"
    combined_clean = clean_text(combined)

    # anomaly score
    X_new = vectorizer.transform([combined_clean])
    raw_df_score = model.decision_function(X_new)[0]
    anomaly_score = normalize_anomaly(
        raw_df_score,
        metadata.get("train_min_score", -1),
        metadata.get("train_max_score", 1),
    )

    # rules
    rule_score, rule_reasons = compute_rule_score_and_reasons(
        combined, file_types, urls, metadata
    )

    # hybrid scoring
    alpha = 0.5  # anomaly weight
    final_score = (alpha * anomaly_score) + ((1 - alpha) * rule_score)

    # ===============================
    # Stricter thresholds
    # ===============================
    if any(x in combined_clean for x in ["password", "verify", "bank", "account", "urgent"]):
        prediction_label = "🚨 Fraud Detected"
        advice = "⚠️ Strong phishing indicators found. Treat as fraud."
        final_score = max(final_score, 0.9)
    elif final_score >= 0.55:
        prediction_label = "🚨 Fraud Detected"
        advice = "⚠️ Likely fraud. Do not trust."
    elif final_score >= 0.3:
        prediction_label = "⚠️ Suspicious - Needs Review"
        advice = "⚠️ Unusual patterns found. Verify manually."
    else:
        prediction_label = "✅ Safe Email"
        advice = "✅ No strong fraud patterns detected."

    # reasons
    reasons = []
    if rule_reasons:
        reasons.append("; ".join(rule_reasons))
    if anomaly_score >= 0.6:
        reasons.append(f"High anomaly score (={round(anomaly_score,3)})")
    reason_text = "; ".join(reasons) if reasons else "No obvious red flags"

    # case type
    case_type = detect_case_type(combined, file_types, urls)

    return {
        "prediction": prediction_label,
        "risk_score": round(final_score * 100, 2),
        "anomaly_score": round(anomaly_score * 100, 2),
        "rule_score": round(rule_score * 100, 2),
        "reasons": reason_text,
        "case_type": case_type,
        "feedback": advice,
        "helpline": {
            "phone": "+1-800-555-FRAUD",
            "email": "reportfraud@cyberhelp.org",
            "address": "Cyber Crime Division, 123 Security Lane, Safe City"
        }
    }

# ===============================
# FEEDBACK SYSTEM
# ===============================
if not os.path.exists(feedback_path):
    pd.DataFrame(columns=["sender_email", "subject", "content", "urls", "file_types", "label"]).to_csv(feedback_path, index=False)

def add_feedback(sender_email, subject, content, urls, file_types, label):
    df = pd.read_csv(feedback_path)
    new = pd.DataFrame([{
        "sender_email": sender_email,
        "subject": subject,
        "content": content,
        "urls": urls,
        "file_types": file_types,
        "label": label
    }])
    df = pd.concat([df, new], ignore_index=True)
    df.to_csv(feedback_path, index=False)
    print(f"✅ Feedback stored: {label}")

def update_model_with_feedback():
    base_df = pd.read_csv(processed_cases_path)
    fb_df = pd.read_csv(feedback_path)

    if not fb_df.empty:
        fb_df["combined"] = (
            fb_df["sender_email"] + " " +
            fb_df["subject"] + " " +
            fb_df["content"] + " " +
            fb_df["urls"] + " " +
            fb_df["file_types"]
        ).apply(clean_text)
        base_df = pd.concat([base_df, fb_df], ignore_index=True)

    new_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=20000)
    X_all = new_vectorizer.fit_transform(base_df["combined"])

    new_model = IsolationForest(contamination=0.08, random_state=42)
    new_model.fit(X_all)

    train_scores = new_model.decision_function(X_all)
    metadata["train_min_score"] = float(np.min(train_scores))
    metadata["train_max_score"] = float(np.max(train_scores))
    metadata["train_mean_score"] = float(np.mean(train_scores))

    joblib.dump(new_model, out_model_path)
    joblib.dump(new_vectorizer, out_vectorizer_path)
    joblib.dump(metadata, out_meta_path)
    print("🔄 Model retrained with feedback")
