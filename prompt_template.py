# ==========================================
# Prompt Templates
# ==========================================

def system_prompt():
    """
    Returns the system prompt that defines
    the chatbot's behavior.
    """

    return """
You are an AI Medical Assistant.

Your responsibilities are:
- Explain disease prediction results.
- Use the retrieved medical knowledge.
- Provide simple and easy-to-understand explanations.
- Recommend consulting a healthcare professional.
- Never replace a doctor's diagnosis.
- Keep responses professional and concise.
"""


# ==========================================
# Prediction Prompt
# ==========================================

def prediction_prompt(patient_note, prediction, retrieved_text):
    """
    Creates the prompt used to explain
    the prediction.
    """

    prompt = f"""
You are an AI Medical Assistant.

Patient Notes:
{patient_note}

Predicted Disease:
{prediction}

Relevant Medical Knowledge:
{retrieved_text}

Generate a clear explanation including:

1. Summary of the patient's symptoms.
2. Why the prediction matches the symptoms.
3. Medical information related to the predicted disease.
4. Lifestyle recommendations.
5. A reminder that this is not a medical diagnosis.
"""

    return prompt


# ==========================================
# Recommendation Prompt
# ==========================================

def recommendation_prompt(prediction):
    """
    Returns disease-specific recommendations.
    """

    if prediction.lower() == "diabetes":

        return """
Recommendations:

• Monitor blood sugar regularly.
• Follow a healthy diet.
• Exercise for at least 30 minutes daily.
• Reduce sugar intake.
• Drink plenty of water.
• Visit an endocrinologist for further evaluation.
"""

    elif prediction.lower() == "heart disease":

        return """
Recommendations:

• Monitor blood pressure regularly.
• Eat a heart-healthy diet.
• Reduce cholesterol intake.
• Exercise regularly.
• Avoid smoking and alcohol.
• Consult a cardiologist for further evaluation.
"""

    else:

        return """
Recommendation:

Please consult a qualified healthcare professional
for further medical evaluation.
"""


# ==========================================
# Disclaimer Prompt
# ==========================================

def disclaimer():

    return """
--------------------------------------------------

DISCLAIMER

This prediction is generated using a Machine Learning
and Retrieval-Augmented Generation (RAG) system.

The result is intended for educational purposes only.

It should not be considered a medical diagnosis
or a substitute for professional healthcare advice.

--------------------------------------------------
"""


# ==========================================
# Example
# ==========================================

if __name__ == "__main__":

    patient_note = "Patient reports chest pain and shortness of breath."

    prediction = "Heart Disease"

    retrieved_text = """
Heart disease affects the heart and blood vessels.

Common symptoms include:
- Chest pain
- Shortness of breath
- Fatigue
- Dizziness
"""

    print(system_prompt())

    print(
        prediction_prompt(
            patient_note,
            prediction,
            retrieved_text
        )
    )

    print(
        recommendation_prompt(
            prediction
        )
    )

    print(disclaimer())
