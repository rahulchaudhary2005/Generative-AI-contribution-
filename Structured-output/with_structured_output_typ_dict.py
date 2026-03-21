from typing import TypedDict
from transformers import pipeline


# =========================
# 🧠 STEP 1: Structured Output
# =========================

class OutputSchema(TypedDict):
    text: str
    sentiment: str
    confidence: float


# =========================
# 🤖 STEP 2: Load Lightweight Model (Local)
# =========================

sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)


# =========================
# 🔗 STEP 3: Analysis Function
# =========================

def analyze_text(user_text: str) -> OutputSchema:
    result = sentiment_model(user_text)[0]

    return {
        "text": user_text,
        "sentiment": result["label"],     # POSITIVE / NEGATIVE
        "confidence": round(result["score"], 3)
    }


# =========================
# ▶️ STEP 4: Run
# =========================

if __name__ == "__main__":
    user_input = input("Enter your text: ")

    output = analyze_text(user_input)

    print("\n===== RESULT =====")
    print(output)