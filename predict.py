import joblib
import numpy as np

# ==========================================
# Load Hybrid Models
# ==========================================

diabetes_model = joblib.load("models/hybrid_diabetes_model.pkl")
heart_model = joblib.load("models/hybrid_heart_model.pkl")

# ==========================================
# Diabetes Prediction
# ==========================================

print("\n========== DIABETES PREDICTION ==========")

pregnancies = float(input("Pregnancies: "))
glucose = float(input("Glucose: "))
blood_pressure = float(input("Blood Pressure: "))
skin_thickness = float(input("Skin Thickness: "))
insulin = float(input("Insulin: "))
bmi = float(input("BMI: "))
dpf = float(input("Diabetes Pedigree Function: "))
age = float(input("Age: "))

diabetes_input = np.array([[
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    dpf,
    age
]])

diabetes_prediction = diabetes_model.predict(diabetes_input)

if diabetes_prediction[0] == 1:
    print("\nResult: High Risk of Diabetes")
else:
    print("\nResult: Low Risk of Diabetes")

# ==========================================
# Heart Disease Prediction
# ==========================================

print("\n========== HEART DISEASE PREDICTION ==========")

age = float(input("Age: "))
sex = float(input("Sex (1=Male, 0=Female): "))
cp = float(input("Chest Pain Type: "))
trestbps = float(input("Resting Blood Pressure: "))
chol = float(input("Cholesterol: "))
fbs = float(input("Fasting Blood Sugar: "))
restecg = float(input("Rest ECG: "))
thalach = float(input("Maximum Heart Rate: "))
exang = float(input("Exercise Induced Angina: "))
oldpeak = float(input("Oldpeak: "))
slope = float(input("Slope: "))
ca = float(input("CA: "))
thal = float(input("Thal: "))

heart_input = np.array([[
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

heart_prediction = heart_model.predict(heart_input)

if heart_prediction[0] == 1:
    print("\nResult: High Risk of Heart Disease")
else:
    print("\nResult: Low Risk of Heart Disease")

print("\nPrediction Completed Successfully!")
