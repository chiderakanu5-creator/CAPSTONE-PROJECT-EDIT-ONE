# -*- coding: utf-8 -*-
"""CAPSTONE PROJECT - IMPROVED MODEL WITH NAIVE BAYES

This script trains a Naive Bayes classifier for spam email detection.
Naive Bayes is better at handling edge cases and gibberish than Random Forest.

Topic: Spam Email Classifiers - Improved Version
"""

import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer

# ==================== 1. IMPORT DATASET ====================
print("Loading dataset...")
df = pd.read_csv('/content/sms_spam.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# ==================== 2. DATA CLEANING ====================
print("\n" + "="*50)
print("DATA CLEANING")
print("="*50)

# Check for missing values
print(f"\nMissing values:\n{df.isnull().sum()}")

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")

# Drop duplicates
df = df.drop_duplicates()
print(f"Dataset shape after removing duplicates: {df.shape}")

# ==================== 3. DATA PREPROCESSING ====================
print("\n" + "="*50)
print("DATA PREPROCESSING")
print("="*50)

# Convert labels to numeric (ham=0, spam=1)
df['type'] = df['type'].map({'ham': 0, 'spam': 1})
df['type'] = df['type'].astype(int)

print(f"\nClass distribution:")
print(df['type'].value_counts())

# ==================== 4. TEXT VECTORIZATION ====================
print("\n" + "="*50)
print("TEXT VECTORIZATION (TF-IDF)")
print("="*50)

vectorizer = TfidfVectorizer(stop_words="english", lowercase=True, max_features=5000)
X = vectorizer.fit_transform(df["text"])
y = df["type"]

print(f"Vectorized data shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")

# ==================== 5. TRAIN-TEST SPLIT ====================
print("\n" + "="*50)
print("TRAIN-TEST SPLIT")
print("="*50)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# ==================== 6. TRAIN NAIVE BAYES MODEL ====================
print("\n" + "="*50)
print("TRAINING NAIVE BAYES CLASSIFIER")
print("="*50)

naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train, y_train)

print("✓ Naive Bayes model trained successfully!")

# ==================== 7. EVALUATE MODEL ====================
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

y_pred = naive_bayes_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)

print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1 Score:  {f1:.4f}")

# ==================== 8. CONFUSION MATRIX ====================
print("\n" + "="*50)
print("CONFUSION MATRIX")
print("="*50)

cm = confusion_matrix(y_test, y_pred)
print(f"\n{cm}")

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])
plt.title("Confusion Matrix - Naive Bayes Classifier")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.show()

# ==================== 9. TEST ON GIBBERISH ====================
print("\n" + "="*50)
print("TEST ON EDGE CASES")
print("="*50)

test_cases = [
    "FDFVOJ[S[OJFO[AESTABFOD]K       FD0FJ=ZD-]KALDS[C",  # Gibberish
    "Hi, can you help me?",  # Legitimate
    "Click here to win $1000!",  # Spam
    "Hello, how are you today?",  # Legitimate
]

print("\nTesting on sample messages:")
for test_message in test_cases:
    transformed = vectorizer.transform([test_message])
    pred = naive_bayes_model.predict(transformed)[0]
    confidence = naive_bayes_model.predict_proba(transformed)[0]
    
    label = "SPAM" if pred == 1 else "HAM"
    conf_score = confidence[pred] * 100
    
    print(f"\n'{test_message}'")
    print(f"  → Prediction: {label} (Confidence: {conf_score:.1f}%)")

# ==================== 10. SAVE MODELS ====================
print("\n" + "="*50)
print("SAVING MODELS")
print("="*50)

joblib.dump(naive_bayes_model, 'naive_bayes_spam_classifier.pkl')
joblib.dump(vectorizer, 'vectorizer_naive_bayes.pkl')

print("✓ Models saved successfully!")
print("  - naive_bayes_spam_classifier.pkl")
print("  - vectorizer_naive_bayes.pkl")

print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)
print("\nNext steps:")
print("1. Download the .pkl files")
print("2. Upload them to your GitHub repository")
print("3. Update app.py to use the new model filenames")
print("4. Redeploy on Streamlit Cloud")
