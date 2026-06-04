import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load("spam_email_classifier_model (1).pkl")
vectorizer = joblib.load("vectorizer (1).pkl")

st.title("Spam Email Classifier")
st.markdown("Enter email text below to classify it as spam or legitimate.")

# Text input area
email_text = st.text_area("Enter Email Text", placeholder="Paste your email here...")

# Predict button
if st.button("Classify Email"):
    if email_text.strip() == "":
        st.warning("Please enter email text to classify.")
    else:
        # Vectorize and predict
        transformed = vectorizer.transform([email_text])
        prediction = model.predict(transformed)
        
        # Display result
        if prediction[0] == 1:
            st.error("🚨 **Spam Detected**")
        else:
            st.success("✅ **Legitimate Email**")
