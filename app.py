import streamlit as st
import joblib
import numpy as np
import pandas as pd

from transformers import BertTokenizer
from transformers import BertForSequenceClassification
import torch

from chatbot.explain_prediction import explain_prediction

# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="AI Medical Diagnosis System",
    page_icon="🩺",
    layout="wide"
)

# =====================================================
# Load Models
# =====================================================

diabetes_model = joblib.load("models/hybrid_diabetes_model.pkl")

heart_model = joblib.load("models/hybrid_heart_model.pkl")

tokenizer = BertTokenizer.from_pretrained(
    "models/bert_model"
)

bert_model = BertForSequenceClassification.from_pretrained(
    "models/bert_model"
)

label_mapping = pd.read_csv(
    "models/label_mapping.csv"
)

labels = label_mapping["Diagnosis"].tolist()

# =====================================================
# Title
# =====================================================

st.title("🩺 AI Medical Diagnosis System")

st.markdown("""
Hybrid Machine Learning + Transformer + RAG Chatbot
""")

# =====================================================
# Sidebar
# =====================================================

menu = st.sidebar.selectbox(

    "Select Module",

    [
        "Diabetes Prediction",
        "Heart Disease Prediction",
        "Medical Text Prediction",
        "RAG Medical Chatbot"
    ]

)

# =====================================================
# Diabetes Prediction
# =====================================================

if menu == "Diabetes Prediction":

    st.header("Diabetes Prediction")

    pregnancies = st.number_input("Pregnancies")

    glucose = st.number_input("Glucose")

    bp = st.number_input("Blood Pressure")

    skin = st.number_input("Skin Thickness")

    insulin = st.number_input("Insulin")

    bmi = st.number_input("BMI")

    dpf = st.number_input("Diabetes Pedigree Function")

    age = st.number_input("Age")

    if st.button("Predict Diabetes"):

        sample = np.array([[

            pregnancies,
            glucose,
            bp,
            skin,
            insulin,
            bmi,
            dpf,
            age

        ]])

        prediction = diabetes_model.predict(sample)

        if prediction[0] == 1:

            st.error("High Risk of Diabetes")

        else:

            st.success("Low Risk of Diabetes")

# =====================================================
# Heart Disease Prediction
# =====================================================

elif menu == "Heart Disease Prediction":

    st.header("Heart Disease Prediction")

    age = st.number_input("Age")

    sex = st.selectbox("Sex",[0,1])

    cp = st.number_input("Chest Pain Type")

    trestbps = st.number_input("Resting Blood Pressure")

    chol = st.number_input("Cholesterol")

    fbs = st.number_input("Fasting Blood Sugar")

    restecg = st.number_input("Rest ECG")

    thalach = st.number_input("Maximum Heart Rate")

    exang = st.selectbox("Exercise Angina",[0,1])

    oldpeak = st.number_input("Old Peak")

    slope = st.number_input("Slope")

    ca = st.number_input("CA")

    thal = st.number_input("Thal")

    if st.button("Predict Heart Disease"):

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

        prediction = heart_model.predict(sample)

        if prediction[0] == 1:

            st.error("High Risk of Heart Disease")

        else:

            st.success("Low Risk of Heart Disease")

# =====================================================
# Medical Text Prediction
# =====================================================

elif menu == "Medical Text Prediction":

    st.header("Medical Text Prediction")

    notes = st.text_area(
        "Enter Patient Notes"
    )

    if st.button("Analyze Notes"):

        inputs = tokenizer(

            notes,

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

        disease = labels[prediction]

        st.success(
            f"Predicted Disease : {disease}"
        )

# =====================================================
# RAG Chatbot
# =====================================================

else:

    st.header("Medical Explanation Chatbot")

    notes = st.text_area(
        "Enter Patient Notes"
    )

    disease = st.selectbox(

        "Predicted Disease",

        [

            "Diabetes",

            "Heart Disease"

        ]

    )

    if st.button("Generate Explanation"):

        response = explain_prediction(

            notes,

            disease

        )

        st.write(response)
