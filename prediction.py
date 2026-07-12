import joblib
import numpy as np
import pandas as pd
import torch

from transformers import (
    BertTokenizer,
    BertForSequenceClassification
)

# ======================================================
# Load Models
# ======================================================

diabetes_model = joblib.load(
    "models/hybrid_diabetes_model.pkl"
)

heart_model = joblib.load(
    "models/hybrid_heart_model.pkl"
)

tokenizer = BertTokenizer.from_pretrained(
    "models/bert_model"
)

bert_model = BertForSequenceClassification.from_pretrained(
    "models/bert_model"
)

bert_model.eval()

# ======================================================
# Load Label Mapping
# ======================================================

label_mapping = pd.read_csv(
    "models/label_mapping.csv"
)

labels = label_mapping["Diagnosis"].tolist()

# ======================================================
# Diabetes Prediction
# ======================================================

def predict_diabetes(
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    dpf,
    age
):

    sample = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    prediction = diabetes_model.predict(sample)[0]

    probability = diabetes_model.predict_proba(sample)[0]

    return {
        "prediction": "High Risk" if prediction == 1 else "Low Risk",
        "confidence": round(max(probability) * 100, 2)
    }


# ======================================================
# Heart Disease Prediction
# ======================================================

def predict_heart(
    age,
    sex,
    cp,
    trestbps,
    chol,
    fbs,
    restecg,
    thalach,
    exang,
    oldpeak,
    slope,
    ca,
    thal
):

    sample = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    prediction = heart_model.predict(sample)[0]

    probability = heart_model.predict_proba(sample)[0]

    return {
        "prediction": "High Risk" if prediction == 1 else "Low Risk",
        "confidence": round(max(probability) * 100, 2)
    }


# ======================================================
# Medical Text Prediction (BERT)
# ======================================================

def predict_medical_text(patient_notes):

    inputs = tokenizer(
        patient_notes,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = bert_model(**inputs)

    prediction = torch.argmax(
        outputs.logits,
        dim=1
    ).item()

    confidence = torch.softmax(
        outputs.logits,
        dim=1
    )[0][prediction].item()

    return {
        "prediction": labels[prediction],
        "confidence": round(confidence * 100, 2)
    }


# ======================================================
# Example
# ======================================================

if __name__ == "__main__":

    diabetes = predict_diabetes(
        2,
        145,
        80,
        25,
        120,
        28.5,
        0.45,
        42
    )

    print("Diabetes Prediction")
    print(diabetes)

    heart = predict_heart(
        58,
        1,
        2,
        140,
        240,
        0,
        1,
        155,
        0,
        1.2,
        2,
        0,
        2
    )

    print("\nHeart Disease Prediction")
    print(heart)

    text = predict_medical_text(
        "Patient reports chest pain, dizziness and shortness of breath."
    )

    print("\nMedical Text Prediction")
    print(text)
