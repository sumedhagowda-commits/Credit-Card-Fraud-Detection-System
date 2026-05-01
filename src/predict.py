import joblib
import numpy as np

def predict_transaction(features):
    model = joblib.load("models/model.pkl")
    prediction = model.predict([features])

    if prediction[0] == 1:
        return "⚠ Fraud Detected"
    else:
        return "✅ Normal Transaction"