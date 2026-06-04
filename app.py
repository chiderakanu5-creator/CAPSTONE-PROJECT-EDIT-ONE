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

# Function to check text quality
def is_valid_text(text):
    """Check if text contains real words (not gibberish)"""
    # Count alphabetic characters vs special characters
    alpha_count = sum(1 for c in text if c.isalpha())
    special_count = sum(1 for c in text if not c.isalnum() and c != ' ')
    total = len(text)
    
    # If more than 30% special characters, it's likely gibberish
    if total > 0 and (special_count / total) > 0.3:
        return False, "⚠️ Text contains too many special characters (likely gibberish)"
    
    # If less than 2 words, might be too short
    words = text.split()
    if len(words) < 2:
        return False, "⚠️ Text is too short to classify"
    
    # If mostly numbers, probably not real text
    digit_count = sum(1 for c in text if c.isdigit())
    if total > 0 and (digit_count / total) > 0.5:
        return False, "⚠️ Text contains too many numbers (not a real message)"
    
    return True, None

st.title("Spam Email Classifier")
st.markdown("Enter email text below to classify it as spam or legitimate.")

model, vectorizer = load_models()

if model and vectorizer:
    email_text = st.text_area("Enter Email Text", placeholder="Paste your email here...")
    
    if st.button("Classify Email"):
        if email_text.strip() == "":
            st.warning("Please enter email text to classify.")
        else:
            # Validate text quality first
            is_valid, error_msg = is_valid_text(email_text)
            
            if not is_valid:
                st.warning(error_msg)
            else:
                # Vectorize and predict
                transformed = vectorizer.transform([email_text])
                prediction = model.predict(transformed)
                confidence = model.predict_proba(transformed)[0]
                
                # Display result with confidence
                if prediction[0] == 1:
                    st.error(f"🚨 **Spam Detected** (Confidence: {confidence[1]:.1%})")
                else:
                    st.success(f"✅ **Legitimate Email** (Confidence: {confidence[0]:.1%})")
