import streamlit as st
import joblib
import re
import os
from PyPDF2 import PdfReader
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# Ensure NLTK resources are available for the application
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
custom_fillers = {'using', 'worked', 'project', 'experience', 'client', 'involved', 'responsibilities', 'role', 'team', 'application', 'system'}
all_stopwords = stop_words.union(custom_fillers)

# Exact preprocessing function matching the training notebook pipeline
def clean_and_anonymize_resume(text):
    text = str(text).lower()
    # Redact raw Emails and Phone Numbers to preserve privacy
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b|\b\d{10}\b', ' ', text)
    # Remove HTML Tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Retain letters, numbers, and technical delimiters (like + and #)
    text = re.sub(r'[^a-zA-Z0-9\s+#\.]', ' ', text)
    tokens = text.split()
    # Lemmatize and filter stopwords
    cleaned_tokens = [
        lemmatizer.lemmatize(word) for word in tokens
        if word not in all_stopwords and (len(word) > 1 or word in ['c', 'r', 'js', 'db'])
    ]
    return ' '.join(cleaned_tokens)

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    try:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""

# Page Config
st.set_page_config(
    page_title="Resume Classification Dashboard",
    page_icon="💼",
    layout="centered"
)

# Main Title & Header
st.title("💼 Resume Classification Dashboard")
st.subheader("AI-Powered Candidate Profile Classifier")
st.write("""
This professional portal automatically parses and classifies resume files into industry-standard profiles
using Natural Language Processing (NLP) and a trained machine learning pipeline.
""")

# Load Model Pipeline
model_path = 'resume_classifier.pkl'
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found! Please make sure it is in the same directory as this script.")
else:
    model = joblib.load(model_path)

    # Layout Tabs
    tab1, tab2 = st.tabs(["📤 Upload Resume File", "✍ Paste Resume Text"])

    with tab1:
        st.markdown("### Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a file (PDF or TXT format only)",
            type=["pdf", "txt"],
            help="Maximum file size supported is 200MB"
        )

        raw_resume_text = ""
        if uploaded_file is not None:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension == 'pdf':
                raw_resume_text = extract_text_from_pdf(uploaded_file)
            elif file_extension == 'txt':
                raw_resume_text = str(uploaded_file.read(), "utf-8")

            if raw_resume_text.strip() == "":
                st.warning("Could not extract meaningful text from the uploaded file.")

    with tab2:
        st.markdown("### Paste Document Text")
        pasted_text = st.text_area(
            "Paste the raw resume text here:",
            height=250,
            placeholder="Paste candidate summary, skillsets, and experience record..."
        )
        if pasted_text.strip() != "":
            raw_resume_text = pasted_text

    # Run Classification Process
    if raw_resume_text.strip() != "":
        st.divider()
        st.markdown("### Processing & Analysis")

        with st.spinner("Parsing, cleaning, and classifying the resume..."):
            # Preprocess the text
            cleaned_text = clean_and_anonymize_resume(raw_resume_text)

            if len(cleaned_text.strip()) == 0:
                st.error("Error: Preprocessed resume is empty! Please verify the input text.")
            else:
                # Perform Prediction
                predicted_class = model.predict([cleaned_text])[0]
                probabilities = model.predict_proba([cleaned_text])[0]
                class_labels = model.classes_

                # Format confidence scores
                conf_dict = dict(zip(class_labels, probabilities))
                predicted_conf = conf_dict[predicted_class] * 100

                # Visual Output Cards
                st.success(f"🎯 **Resume Successfully Classified!**")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Predicted Profile Category", value=predicted_class)
                with col2:
                    st.metric(label="Prediction Confidence Score", value=f"{predicted_conf:.2f}%")

                # Display Probability breakdown as a clean horizontal chart
                st.write("#### Full Class Distribution Breakdown:")
                for label, prob in conf_dict.items():
                    st.progress(float(prob), text=f"{label}: {prob*100:.2f}%")

                # Show Sanitized View (Demonstrates Anonymization & Extraction Integrity)
                with st.expander("🔍 View Sanitized NLP Representation (Emails & Phones Removed)"):
                    st.write(cleaned_text)

st.markdown("--- ")
st.markdown("💡 *This ML service supports four key target classifications: React Developer, Workday, Peoplesoft Resume, and SQL Developer.*")
