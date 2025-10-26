import os
import joblib
from django.conf import settings

# Load ML model and vectorizer once when Django starts
MODEL_PATH = os.path.join(settings.BASE_DIR, "models/fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(settings.BASE_DIR, "models/tfidf_vectorizer.pkl")

model = None
vectorizer = None

def load_model():
    global model, vectorizer
    if model is None or vectorizer is None:
        print("[INFO] Loading ML model and TF-IDF vectorizer...")
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict_news(text):
    """Predict whether a given news text is Fake or Real."""
    model, vectorizer = load_model()
    transformed = vectorizer.transform([text])
    prediction = model.predict(transformed)[0]
    prob = model.predict_proba(transformed)[0].max() * 100
    label = "Real" if prediction == 1 else "Fake"
    return label, round(prob, 2)
