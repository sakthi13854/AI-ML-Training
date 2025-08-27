import os
import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from PyPDF2 import PdfReader

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

EMBED_MODEL = "text-embedding-004"
PRIMARY_LLM_MODEL = "gemini-1.5-pro"
FALLBACK_LLM_MODEL = "gemini-1.5-flash"

def chunk_text(text, size=500, overlap=100):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+size]))
        i += size - overlap
    return chunks

def embed(texts, task_type="retrieval_document"):
    embeddings = []
    for t in texts:
        resp = genai.embed_content(
            model=EMBED_MODEL,
            content=t,
            task_type=task_type
        )
        embeddings.append(resp["embedding"])
    return np.array(embeddings, dtype="float32")

def embed_query(query):
    resp = genai.embed_content(
        model=EMBED_MODEL,
        content=query,
        task_type="retrieval_query"
    )
    return np.array(resp["embedding"], dtype="float32")[None, :]

def read_file(uploaded_file):
    if uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")
    elif uploaded_file.name.endswith(".pdf"):
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text
    return ""

@st.cache_resource
def build_index(files):
    docs, chunks = [], []
    for uploaded_file in files:
        text = read_file(uploaded_file)
        for c in chunk_text(text):
            docs.append({"text": c, "source": uploaded_file.name})
            chunks.append(c)
    X = embed(chunks)
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)
    return index, docs

def retrieve(index, docs, query, k=3):
    qvec = embed_query(query)
    D, I = index.search(qvec, k)
    return [docs[i] for i in I[0]]

def answer(query, passages):
    context = "\n\n".join([f"Source: {p['source']}\n{p['text']}" for p in passages])
    prompt = f"Answer the question using only the sources.\n\nQuestion: {query}\n\nSources:\n{context}\n\nAnswer:"
    try:
        model = genai.GenerativeModel(PRIMARY_LLM_MODEL)
        resp = model.generate_content(prompt)
        return resp.text
    except ResourceExhausted:
        model = genai.GenerativeModel(FALLBACK_LLM_MODEL)
        resp = model.generate_content(prompt)
        return f"(⚡ Fallback to {FALLBACK_LLM_MODEL})\n\n" + resp.text

st.set_page_config(page_title="Gemini RAG Chatbot", page_icon="=")
st.title("Gemini RAG Chatbot")
st.caption("Upload your custom dataset (.txt / .pdf) and chat with Gemini + RAG")

uploaded_files = st.file_uploader("Upload text or PDF files", type=["txt", "pdf"], accept_multiple_files=True)

if uploaded_files:
    if "index" not in st.session_state:
        with st.spinner("Building vector index..."):
            st.session_state.index, st.session_state.docs = build_index(uploaded_files)
        st.success("Index built! You can now chat with your data.")

    query = st.text_input("Ask a question about your documents:")
    if st.button("Ask") and query.strip():
        with st.spinner("Thinking..."):
            passages = retrieve(st.session_state.index, st.session_state.docs, query, k=3)
            ans = answer(query, passages)
        st.markdown("**Answer**")
        st.write(ans)
        st.markdown("**Sources**")
        for i, p in enumerate(passages, 1):
            st.write(f"[{i}] **{p['source']}** — {p['text'][:150]}...")
else:
    st.info("Upload one or more `.txt` or `.pdf` files to start chatting.")
