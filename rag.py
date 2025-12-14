
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from openai import OpenAI

# ---------------------------
# Streamlit Config
# ---------------------------
st.set_page_config(
    page_title="RAG Document Search (GPT-4o-mini)",
    layout="wide"
)
st.title("Document Search & Summarization (RAG + GPT-4o-mini)")

# ---------------------------
# Sidebar – API Key
# ---------------------------
st.sidebar.header("OpenAI API Configuration")
api_key = st.sidebar.text_input(
    "Enter your OpenAI API Key",
    type="password"
)

if not api_key:
    st.warning("Please enter your OpenAI API key")
    st.stop()

client = OpenAI(api_key=api_key)

# ---------------------------
# Sidebar – Summary Settings
# ---------------------------
st.sidebar.header("Summary Settings")
SUMMARY_TOKENS = st.sidebar.slider(
    "Summary length (tokens)",
    100, 1000, 250
)

# ---------------------------
# File Upload (Single File)
# ---------------------------
st.subheader("Upload a Document (PDF or TXT)")

uploaded_file = st.file_uploader(
    "Upload ONE PDF or TXT file",
    type=["pdf", "txt"],
    accept_multiple_files=False
)

if not uploaded_file:
    st.info("Please upload a PDF or TXT file to continue.")
    st.stop()

# ---------------------------
# Load Embedding Model
# ---------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------
# Parse File
# ---------------------------
def parse_uploaded_file(file):
    chunks = []

    if file.type == "text/plain":
        text = file.read().decode("utf-8")
        chunks = [t.strip() for t in text.split("\n\n") if t.strip()]

    elif file.type == "application/pdf":
        reader = PdfReader(file)
        full_text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                full_text += extracted + "\n"
        chunks = [t.strip() for t in full_text.split("\n\n") if t.strip()]

    return chunks

documents = parse_uploaded_file(uploaded_file)

if len(documents) == 0:
    st.error("No readable text found in the uploaded document.")
    st.stop()

# ---------------------------
# Build Indexes
# ---------------------------
with st.spinner("Indexing document..."):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)

    embedding_model = load_embedding_model()
    embeddings = embedding_model.encode(documents)

st.success(f"Indexed {len(documents)} sections from the document")

# ---------------------------
# Search
# ---------------------------
query = st.text_input("Enter your query")

if query:
    q_vec = vectorizer.transform([query])
    tfidf_scores = cosine_similarity(q_vec, tfidf_matrix)[0]

    q_emb = embedding_model.encode([query])
    emb_scores = cosine_similarity(q_emb, embeddings)[0]

    final_scores = 0.5 * tfidf_scores + 0.5 * emb_scores
    best_idx = int(np.argmax(final_scores))

    best_chunk = documents[best_idx]
    best_score = final_scores[best_idx]

    # ---------------------------
    # Show Best Match
    # ---------------------------
    st.subheader("Most Relevant Section")
    with st.expander(f"Best Match (Score: {best_score:.3f})"):
        st.write(best_chunk)

    # ---------------------------
    # GPT-4o-mini Summary
    # ---------------------------
    st.subheader("GPT-4o-mini Summary")
    with st.spinner("Summarizing..."):
        prompt = f"""
Summarize the following content clearly and concisely.
Focus on the key ideas and important details.

CONTENT:
{best_chunk}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes documents."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=SUMMARY_TOKENS,
            temperature=0.3
        )

        summary = response.choices[0].message.content
        st.write(summary)

st.markdown("---")
st.caption(
    "Single-document RAG | TF-IDF + Sentence Embeddings | GPT-4o-mini"
)
