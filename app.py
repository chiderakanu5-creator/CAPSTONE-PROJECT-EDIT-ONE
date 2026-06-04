import streamlit as st
import joblib
import os

@st.cache_resource
def load_models():
    try:
        model = joblib.load("spam_email_classifier_model (1).pkl")
        vectorizer = joblib.load("vectorizer (1).pkl")
        return model, vectorizer
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Ensure .pkl files are in the app directory.")
        return None, None

st.title("Spam Email Classifier")
st.markdown("Enter email text below to classify it as spam or legitimate.")

model, vectorizer = load_models()

if model and vectorizer:
    email_text = st.text_area("Enter Email Text", placeholder="Paste your email here...")
    
    if st.button("Classify Email"):
        if email_text.strip() == "":
            st.warning("Please enter email text to classify.")
        else:
            transformed = vectorizer.transform([email_text])
            prediction = model.predict(transformed)
            confidence = model.predict_proba(transformed)[0]
            
            if prediction[0] == 1:
                st.error(f"🚨 **Spam Detected** (Confidence: {confidence[1]:.1%})")
            else:
                st.success(f"✅ **Legitimate Email** (Confidence: {confidence[0]:.1%})")
