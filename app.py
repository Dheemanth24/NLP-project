import streamlit as st
import pickle
import re

# Page configurations
st.set_page_config(page_title="AI Spam Detector", page_icon="🛡️")

# Custom CSS to inject a modern dark-theme background gradient
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #111827 0%, #1a103c 50%, #030712 100%);
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🛡️ AI Email & SMS Spam Classifier")
st.write("Type a message below to see if the trained model flags it as Spam or Safe.")

# Load backend files from model_new.py
@st.cache_resource
def load_model():
    with open('spam_sms_model.pkl', 'rb') as f:
        return pickle.load(f)

# Basic text cleaning pipeline
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Lowercase conversion
    text = text.lower()
    
    # Swap raw urls out so the vector weights can track them better
    text = re.sub(r"http\S+|www\.\S+", "linkplaceholder", text)
    
    # Strip symbols and collapse spaces
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Unpack pickles or kill process if missing
try:
    vectorizer, model = load_model()
except Exception:
    st.error("Error loading model file.")
    st.stop()

# Wrapping in a form allows the user to press 'Enter' on their keyboard to run it
with st.form("spam_checker_form", clear_on_submit=False):
    
    user_input = st.text_area(
        "Enter the message text here:", 
        height=150, 
        placeholder="e.g., Congratulations! You have won a free prize..."
    )
    
    # The form submit button acts as our analysis trigger
    submit_button = st.form_submit_button("Analyze Message", type="primary")

# Execute logic if the button is clicked OR if the user hits Enter inside the text block
if submit_button:
    if user_input.strip() != "":
        # Preprocess string
        cleaned_text = clean_text(user_input)
        
        # Numeric extraction via trained vocab weights
        vectorized_text = vectorizer.transform([cleaned_text])
        
        # Calculate matrix array for label scores [P(ham), P(spam)]
        probabilities = model.predict_proba(vectorized_text)[0][1]
        spam_probability = probabilities
        
        st.write("---")
        
        # Strict boundary threshold check for borderline stock/phishing edge-cases
        if spam_probability > 0.15:
            st.error(f"🚨 **Warning: This looks like a SPAM message!** (Suspicion Score: {spam_probability * 100:.1f}%)")
        else:
            st.success(f"✅ **Safe: This looks like a legitimate message (HAM).** (Suspicion Score: {spam_probability * 100:.1f}%)")
            
        # Terminal diagnostic logs for debugging
        st.caption(f"Backend diagnostics: Cleaned input evaluated as '{cleaned_text}'")
    else:
        st.warning("Please type a message first before running the analysis!")
