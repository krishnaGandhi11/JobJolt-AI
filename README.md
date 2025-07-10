# 🧠 Resume Role Classifier (ML Project)

This project uses **Machine Learning and Natural Language Processing** to automatically predict the most suitable job role for a candidate based on their resume content.

## 🔍 Problem Statement
Many candidates apply for the wrong job roles, or recruiters have to manually filter hundreds of resumes. This model solves that by **classifying resumes** into roles like:
- Data Scientist
- Python Developer
- Java Developer
- DevOps Engineer
- HR, Sales, etc.

## 📁 Project Structure

ResumeRoll/
├── data/
│ └── UpdatedResumeDataSet.csv
├── model/
│ ├── resume_classifier_model.pkl
│ └── tfidf_vectorizer.pkl
├── model.py ← Trains and saves the model
├── app.py ← Streamlit app to classify resume text
├── requirements.txt ← All required Python packages
├── README.md

## ⚙️ How to Run

1. Clone the repo or download the files.
2. Install requirements: pip install -r requirements.txt
3. Run the Streamlit app: streamlit run app.py

4. Paste any resume text in the textbox and click **Predict** to get the job role!

## 📊 Model Info

- **Vectorization:** TF-IDF (3000 features)
- **Model Used:** Multinomial Naive Bayes
- **Accuracy:** ~98.4%
- **Libraries Used:** pandas, scikit-learn, streamlit, nltk

## ✨ Features

- Cleans and pre-processes resume data
- Converts text to numerical features
- Predicts 25+ job roles using a trained ML model
- Simple UI via Streamlit

## 👨‍💻 Developed By

Krishna Gandhi  
Machine Learning using Python - Summer Training 2025

---

⭐️ Star this repo if you like it!
