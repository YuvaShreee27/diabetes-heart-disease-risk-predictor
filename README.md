
# Diabetes / Heart Disease Risk Predictor
This project is used to predict whether a person is at risk of Diabetes or Heart Disease using Machine Learning algorithms.
## Week 1 Task
Logistic Regression + Random Forest on Medical Dataset
## Project Objective
- Build a machine learning model to predict disease risk.
- Train and compare Logistic Regression and Random Forest algorithms.
- Evaluate model performance using accuracy.
- Save the best-performing model for future deployment.
## Tools Used
- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
## Machine Learning Algorithms
- Logistic Regression
- Random Forest Classifier
## Week 1 Progress
- Collected medical dataset
- Loaded and preprocessed the dataset
- Trained Logistic Regression model
- Trained Random Forest model
- Compared model accuracy
- Saved the trained model
## Future Work
- Build a web application using Flask
- Improve prediction accuracy
- Deploy the project online
- Add user-friendly interface
## Week 2 - Feature Engineering, Model Training & Performance Evaluation
### Project Description
This project predicts the risk of Diabetes and Heart Disease using Machine Learning. In Week 2, the datasets were preprocessed, machine learning models were trained, and their performance was evaluated.
---
## Objectives
- Clean and preprocess the datasets.
- Train machine learning models.
- Evaluate the models using basic performance metrics.
---
## Tasks Completed
- Loaded the Diabetes and Heart Disease datasets.
- Checked for missing values.
- Removed duplicate records.
- Scaled the features using StandardScaler.
- Trained Logistic Regression model.
- Trained Random Forest model.
- Evaluated the models using Accuracy Score, Confusion Matrix, and Classification Report.
---
## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
---
## Files Included
```
feature_engineering.py
model_training.py
performance_metrics.py
diabetes.csv
heart.csv
diabetes_cleaned.csv
heart_cleaned.csv
diabetes_logistic_model.pkl
diabetes_randomforest_model.pkl
heart_logistic_model.pkl
heart_randomforest_model.pkl
Week2_Report.pdf
README.md
```
---
## Performance Metrics
- Accuracy Score
- Confusion Matrix
- Classification Report
---
## Outcome
- Successfully cleaned and preprocessed both datasets.
- Trained Logistic Regression and Random Forest models.
- Evaluated the models using standard performance metrics.
- Saved the cleaned datasets and trained models for future use.
---
## Conclusion
The Week 2 tasks were completed successfully. The datasets were cleaned, machine learning models were trained, and their performance was evaluated. This improved the overall Diabetes and Heart Disease Risk Predictor project.
# Week 3 – Hybrid Disease Risk Prediction Model
## Project Title
**Diabetes & Heart Disease Risk Predictor**
## Objective
The objective of Week 3 was to improve the prediction performance of the existing machine learning models by developing a **Hybrid Model**. A Voting Classifier was implemented by combining **Logistic Regression** and **Random Forest** algorithms to achieve more reliable and accurate disease risk prediction.
---
## Work Completed
* Used the cleaned diabetes and heart disease datasets from Week 2.
* Developed a Hybrid Machine Learning Model using a Voting Classifier.
* Combined Logistic Regression and Random Forest classifiers.
* Trained separate hybrid models for diabetes and heart disease prediction.
* Evaluated model performance using standard classification metrics.
* Saved the trained models for future deployment.
---
## Technologies Used
* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib
---
## Project Structure
```text
Week-3/
│
├── datasets/
│   ├── diabetes_cleaned.csv
│   └── heart_cleaned.csv
│
├── notebooks/
│   └── hybrid_model.ipynb
│
├── models/
│   ├── hybrid_diabetes_model.pkl
│   └── hybrid_heart_model.pkl
│
├── src/
│   ├── hybrid_model.py
│   ├── train_hybrid.py
│   └── evaluate_model.py
│
├── results/
│   ├── accuracy_graph.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── README.md
└── Week3_Report.pdf
```
---
## Model Performance (Sample)
| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 84%      |
| Random Forest       | 89%      |
| **Hybrid Model**    | **91%**  |
*Note: The above values are sample results for demonstration purposes.*
---
## Learning Outcomes
* Understood the concept of ensemble learning.
* Learned how a Voting Classifier combines multiple models to improve prediction performance.
* Gained experience in training, evaluating, and saving hybrid machine learning models.
* Improved knowledge of model comparison using performance metrics.
---
## Future Work
* Integrate Transformer-based models for processing clinical or patient text data.
* Build an explainable AI module for disease prediction.
* Deploy the hybrid prediction model as a web application.
* Enhance the system with additional healthcare datasets for improved accuracy.
# Week 4 – Medical Text Processing Using Transformers
## Project Title
**Diabetes & Heart Disease Risk Predictor**
## Objective
The objective of Week 4 was to explore **Transformer-based Natural Language Processing (NLP)** techniques for processing patient notes and medical text. A pre-trained transformer model was used to analyze clinical text, extract meaningful information, and support disease risk prediction.
---
## Work Completed
* Studied the basics of Transformer architecture for NLP.
* Used a pre-trained transformer model for medical text processing.
* Loaded and processed sample patient notes.
* Performed text preprocessing and tokenization.
* Generated contextual text embeddings for clinical information.
* Integrated transformer-based text features into the disease prediction workflow.
* Evaluated the effectiveness of text processing for supporting disease risk analysis.
---
## Technologies Used
* Python
* Transformers (Hugging Face)
* PyTorch
* Pandas
* NumPy
* Scikit-learn
---
## Project Structure
```text
Week-4/
│
├── datasets/
│   └── patient_notes.csv
│
├── notebooks/
│   └── transformer_medical_text.ipynb
│
├── models/
│   └── transformer_model/
│
├── src/
│   ├── preprocess_text.py
│   ├── transformer_model.py
│   └── predict_text.py
│
├── results/
│   ├── text_embeddings.png
│   ├── prediction_results.png
│   └── model_output.txt
│
├── README.md
└── Week4_Report.pdf
```
---
## Model Performance (Sample)
| Task                            | Result |
| ------------------------------- | ------ |
| Medical Text Classification     | 90%    |
| Text Processing Accuracy        | 92%    |
| Transformer Prediction Accuracy | 91%    |
*Note: The above values are sample results for demonstration purposes.*
---
## Learning Outcomes
* Understood the fundamentals of Transformer-based NLP.
* Learned how tokenization and attention mechanisms improve text understanding.
* Gained experience in processing medical text using pre-trained transformer models.
* Learned how NLP can complement structured medical data for disease prediction.
---
## Future Work
* Develop a Retrieval-Augmented Generation (RAG) chatbot for explaining disease predictions.
* Combine structured patient data and medical text into a unified prediction system.
* Improve model performance using larger clinical datasets.
* Deploy the complete AI-powered disease prediction system as a web application.
# Week 5 – RAG-Based Medical Explanation Chatbot & Web Deployment
## Project Title
**Diabetes & Heart Disease Risk Predictor**
## Objective
The objective of Week 5 was to develop a **Retrieval-Augmented Generation (RAG)** chatbot that provides explanations for disease risk predictions. The final machine learning model was integrated into a web application, allowing users to receive predictions along with AI-generated medical explanations through an interactive interface.
---
## Work Completed
* Studied the fundamentals of Retrieval-Augmented Generation (RAG).
* Developed an AI chatbot to explain diabetes and heart disease prediction results.
* Connected the chatbot with the trained disease prediction models.
* Retrieved relevant medical information to generate user-friendly explanations.
* Built a simple web interface for prediction and chatbot interaction.
* Tested the complete system with sample patient data.
* Prepared the final integrated model for deployment.
---
## Technologies Used
* Python
* Scikit-learn
* Transformers
* LangChain
* FAISS
* Flask
* HTML
* CSS
* JavaScript
---
## Project Structure
```text
Week-5/
│
├── datasets/
│   ├── diabetes_cleaned.csv
│   └── heart_cleaned.csv
│
├── models/
│   ├── hybrid_diabetes_model.pkl
│   └── hybrid_heart_model.pkl
│
├── chatbot/
│   ├── rag_chatbot.py
│   ├── retrieval.py
│   └── prompt_template.py
│
├── web_app/
│   ├── app.py
│   ├── templates/
│   ├── static/
│   └── requirements.txt
│
├── results/
│   ├── chatbot_output.png
│   ├── web_interface.png
│   └── prediction_results.png
│
├── README.md
└── Week5_Report.pdf
```
---
## Final Model Performance (Sample)
| Component                          | Result |
| ---------------------------------- | ------ |
| Hybrid Disease Prediction Accuracy | 91%    |
| Medical Text Processing Accuracy   | 92%    |
| RAG Chatbot Response Accuracy      | 90%    |
| Overall System Performance         | 91%    |

*Note: The above values are sample results for demonstration purposes.*
---
## Learning Outcomes
* Understood the working principles of Retrieval-Augmented Generation (RAG).
* Learned how AI chatbots retrieve and generate informative responses.
* Gained experience in integrating machine learning models with a web application.
* Improved knowledge of deploying an end-to-end healthcare prediction system.
* Developed skills in building user-friendly AI applications.
---
## Future Work
* Improve chatbot responses using larger medical knowledge bases.
* Integrate electronic health record (EHR) support.
* Enhance prediction accuracy with larger datasets and advanced deep learning models.
* Deploy the application on a cloud platform for public access.
* Add multilingual support for better accessibility.

