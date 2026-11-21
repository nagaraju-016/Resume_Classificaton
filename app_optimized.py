import os
import re

import joblib
import nltk
import streamlit as st
from PyPDF2 import PdfReader
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ---------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Resume Classification Dashboard",
    page_icon="💼",
    layout="wide",
)


# ---------------------------------------------------------
# Cached resource initialization
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Preparing NLP resources...")
def load_nlp_resources():
    """Download NLTK data only when missing, then cache NLP objects."""
    required_resources = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
    }

    for resource_path, package_name in required_resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            success = nltk.download(package_name, quiet=True)
            if not success:
                raise RuntimeError(
                    f"Could not download the NLTK resource: {package_name}"
                )

    # WordNet may also require the Open Multilingual Wordnet package
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4", quiet=True)

    return WordNetLemmatizer(), set(stopwords.words("english"))


@st.cache_resource(show_spinner="Loading resume classification model...")
def load_model(model_path):
    """Load and cache the trained model."""
    return joblib.load(model_path)


# ---------------------------------------------------------
# Text preprocessing
# Keep this consistent with the training notebook.
# ---------------------------------------------------------
CUSTOM_FILLERS = {
    "using", "worked", "project", "experience", "client",
    "involved", "responsibilities", "role", "team",
    "application", "system",
}


def clean_and_anonymize_resume(text, lemmatizer, stop_words):
    text = str(text).lower()

    # Redact email addresses and phone numbers
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b|\b\d{10}\b", " ", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Retain letters, numbers, spaces, +, # and periods
    text = re.sub(r"[^a-zA-Z0-9\s+#\.]", " ", text)

    tokens = text.split()
    all_stopwords = stop_words.union(CUSTOM_FILLERS)

    cleaned_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in all_stopwords
        and (len(word) > 1 or word in ["c", "r", "js", "db"])
    ]

    return " ".join(cleaned_tokens)


# ---------------------------------------------------------
# PDF / TXT extraction
# ---------------------------------------------------------
def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PdfReader(uploaded_file)

        if pdf_reader.is_encrypted:
            try:
                pdf_reader.decrypt("")
            except Exception:
                raise ValueError("This PDF is password-protected.")

        extracted_pages = []
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_pages.append(page_text)

        return "\n".join(extracted_pages)

    except Exception as error:
        raise ValueError(f"Could not read this PDF: {error}") from error


def read_uploaded_resume(uploaded_file):
    extension = uploaded_file.name.rsplit(".", 1)[-1].lower()

    if extension == "pdf":
        return extract_text_from_pdf(uploaded_file)

    if extension == "txt":
        try:
            return uploaded_file.getvalue().decode("utf-8")
        except UnicodeDecodeError:
            return uploaded_file.getvalue().decode("utf-8", errors="replace")

    raise ValueError("Please upload a PDF or TXT file.")


# ---------------------------------------------------------
# Load model and NLP resources
# ---------------------------------------------------------
MODEL_PATH = "resume_classifier.pkl"

st.title("💼 Resume Classification Dashboard")
st.subheader("AI-Powered Candidate Profile Classifier")
st.write(
    "Upload a resume or paste resume text to classify the candidate profile "
    "using your trained machine learning model."
)

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file '{MODEL_PATH}' was not found. "
        "Place it in the same folder as app.py."
    )
    st.stop()

try:
    model = load_model(MODEL_PATH)
    lemmatizer, stop_words = load_nlp_resources()
except Exception as error:
    st.error(f"Application initialization failed: {error}")
    st.stop()


# ---------------------------------------------------------
# Resume input
# ---------------------------------------------------------
tab_upload, tab_paste = st.tabs(
    ["📤 Upload Resume File", "✍ Paste Resume Text"]
)

raw_resume_text = ""

with tab_upload:
    st.markdown("### Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT resume",
        type=["pdf", "txt"],
        help="Upload a text-based PDF or TXT file.",
        key="resume_file",
    )

    if uploaded_file is not None:
        # Streamlit's upload limit is also controlled by server configuration.
        if uploaded_file.size > 200 * 1024 * 1024:
            st.error("The uploaded file exceeds the 200 MB application limit.")
        else:
            try:
                with st.spinner("Extracting resume text..."):
                    raw_resume_text = read_uploaded_resume(uploaded_file)
                if not raw_resume_text.strip():
                    st.warning(
                        "No readable text was extracted. "
                        "If this is a scanned PDF, OCR may be required."
                    )
            except Exception as error:
                st.error(str(error))

with tab_paste:
    st.markdown("### Paste Resume Text")
    pasted_text = st.text_area(
        "Paste the raw resume text here:",
        height=250,
        placeholder="Paste candidate summary, skills, education and experience...",
        key="pasted_resume_text",
    )

    if pasted_text.strip():
        raw_resume_text = pasted_text


# ---------------------------------------------------------
# Classification
# ---------------------------------------------------------
if raw_resume_text.strip():
    st.divider()
    st.markdown("### Processing & Analysis")

    try:
        with st.spinner("Cleaning and classifying the resume..."):
            cleaned_text = clean_and_anonymize_resume(
                raw_resume_text,
                lemmatizer,
                stop_words,
            )

            if not cleaned_text.strip():
                st.error(
                    "The preprocessed resume is empty. "
                    "Please check the supplied resume text."
                )
                st.stop()

            predicted_class = model.predict([cleaned_text])[0]

            if not hasattr(model, "predict_proba"):
                st.error(
                    "This trained model does not support predict_proba(), "
                    "so confidence distribution cannot be displayed."
                )
                st.stop()

            probabilities = model.predict_proba([cleaned_text])[0]
            class_labels = model.classes_

        confidence_dict = dict(zip(class_labels, probabilities))
        predicted_confidence = confidence_dict[predicted_class] * 100

        st.success("🎯 Resume successfully classified!")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Predicted Profile Category",
                str(predicted_class),
            )
        with col2:
            st.metric(
                "Prediction Confidence Score",
                f"{predicted_confidence:.2f}%",
            )

        st.markdown("#### Full Class Distribution Breakdown")
        for label, probability in sorted(
            confidence_dict.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            st.progress(
                float(probability),
                text=f"{label}: {probability * 100:.2f}%",
            )

        with st.expander(
            "🔍 View Sanitized NLP Representation (Emails & Phones Removed)"
        ):
            st.write(cleaned_text)

        st.caption(
            "Note: Prediction confidence is the model's probability output; "
            "it is not necessarily a measure of real-world accuracy."
        )

    except Exception as error:
        st.error(f"Classification failed: {error}")


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()
st.caption(
    "Supported target categories in this project: React Developer, "
    "Workday, Peoplesoft Resume, and SQL Developer."
)
