import streamlit as st

from chatbot.explain_prediction import explain_prediction

# ==========================================
# Chatbot User Interface
# ==========================================

def chatbot_interface():

    st.title("🩺 AI Medical Explanation Chatbot")

    st.write(
        """
        Ask the chatbot about your prediction and receive
        an explanation based on the medical knowledge base.
        """
    )

    patient_notes = st.text_area(
        "Enter Patient Notes",
        height=150,
        placeholder="Example: Patient reports chest pain and shortness of breath."
    )

    prediction = st.selectbox(
        "Predicted Disease",
        [
            "Diabetes",
            "Heart Disease"
        ]
    )

    if st.button("Generate Explanation"):

        if patient_notes.strip() == "":

            st.warning("Please enter patient notes.")

        else:

            explanation = explain_prediction(
                patient_notes,
                prediction
            )

            st.success("Explanation Generated Successfully!")

            st.markdown("### Medical Explanation")

            st.text(explanation)

# ==========================================
# Main
# ==========================================

if __name__ == "__main__":

    chatbot_interface()
  
