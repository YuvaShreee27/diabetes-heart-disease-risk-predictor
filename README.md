
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

