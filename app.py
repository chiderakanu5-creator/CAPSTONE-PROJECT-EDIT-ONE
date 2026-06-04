import streamlit as st
import joblib
import os

@st.cache_resource
def load_models():
    try:
        # Load the improved Naive Bayes model
        model = joblib.load("naive_bayes_spam_classifier.pkl")
        vectorizer = joblib.load("vectorizer_naive_bayes.pkl")
        return model, vectorizer
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Ensure .pkl files are in the app directory.")
        return None, None

# Function to check for EXTREME gibberish only (pure random characters)
def is_extreme_gibberish(text):
    """Only reject PURE gibberish (>50% special chars AND <2 words)"""
    special_count = sum(1 for c in text if not c.isalnum() and c != ' ')
    total = len(text)
    
    # Only reject if MOSTLY special characters (>50%)
    if total > 0 and (special_count / total) > 0.5:
        # AND has very few actual words
        words = text.split()
        if len(words) < 2:
            return True
    
    return False

st.title("Spam Email Classifier")
st.markdown("Enter email text below to classify it as spam or legitimate.")
st.markdown("*Using improved Naive Bayes model for better accuracy*")

model, vectorizer = load_models()

if model and vectorizer:
    email_text = st.text_area("Enter Email Text", placeholder="Paste your email here...")
    
    if st.button("Classify Email"):
        if email_text.strip() == "":
            st.warning("Please enter email text to classify.")
        else:
            # Only reject EXTREME gibberish
            if is_extreme_gibberish(email_text):
                st.warning("⚠️ Input appears to be gibberish. Please enter a real message.")
            else:
                # Vectorize and predict
                transformed = vectorizer.transform([email_text])
                prediction = model.predict(transformed)
                confidence = model.predict_proba(transformed)[0]
                
                # Display result with confidence
                if prediction[0] == 1:
                    st.error(f"🚨 **SPAM DETECTED** (Confidence: {confidence[1]:.1%})")
                else:
                    st.success(f"✅ **LEGITIMATE EMAIL** (Confidence: {confidence[0]:.1%})")
