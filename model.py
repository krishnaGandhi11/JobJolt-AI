# ======================== Imports ========================
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

import joblib  # For saving model and vectorizer

# ======================== Data Loading ========================
# Load the dataset
df = pd.read_csv(r"C:\Users\krish\Desktop\UpdatedResumeDataSet.csv", encoding='latin1')

print("Sample data preview:")
print(df.head())

print("\nCategory distribution:")
print(df['Category'].value_counts())

# ======================== Text Preprocessing ========================
# Download required NLTK resources (run once if not already downloaded)
# nltk.download('stopwords')
# nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_resume(text):
    """
    Cleans the resume text by removing URLs, emails, special characters, numbers,
    converting to lowercase, removing stopwords, and lemmatizing.
    """
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply text cleaning
df['Cleaned_Resume'] = df['Resume'].apply(clean_resume)

# ======================== Feature Extraction ========================
# Create TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=3000)

# Transform the cleaned text data
X = tfidf.fit_transform(df['Cleaned_Resume']).toarray()
y = df['Category']

print("\nTF-IDF Matrix Shape:", X.shape)

# ======================== Model Training ========================
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# ======================== Model Evaluation ========================
# Predict on test set
y_pred = model.predict(X_test)

# Print evaluation metrics
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ======================== Save Model & Vectorizer ========================
# Save the trained model and vectorizer as .pkl files
joblib.dump(model, 'resume_classifier_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
