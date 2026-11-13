
import streamlit as st
import torch
import zipfile
import requests
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection System")
st.write("Analyze a news article using our fine-tuned DistilBERT model.")

# Replace this with your GitHub Release asset download URL
MODEL_URL = "https://github.com/PrernaAggarwal/fake-news-detection/releases/download/v1.0/best_distilbert.zip"

MODEL_DIR = "best_distilbert"
ZIP_PATH = "best_distilbert.zip"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_DIR):
        response = requests.get(MODEL_URL, timeout=300)
        response.raise_for_status()

        with open(ZIP_PATH, "wb") as f:
            f.write(response.content)

        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(MODEL_DIR)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    return tokenizer, model

with st.spinner("Loading AI model..."):
    tokenizer, model = load_model()

article = st.text_area(
    "Paste your news article below:",
    height=220,
    placeholder="Enter the news article here..."
)

if st.button("Analyze News", use_container_width=True):
    if not article.strip():
        st.warning("Please enter a news article first.")
    else:
        inputs = tokenizer(
            article,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)[0]

        prediction = torch.argmax(probabilities).item()
        confidence = probabilities[prediction].item() * 100

        if prediction == 0:
            st.error("Prediction: FAKE NEWS")
        else:
            st.success("Prediction: REAL NEWS")

        st.metric("Model confidence", f"{confidence:.2f}%")
        st.caption(
            "This is an AI-generated prediction, not a guarantee "
            "that the article is true or false."
        )
