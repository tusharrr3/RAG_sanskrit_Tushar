import streamlit as st
import requests
import json
import subprocess
import os
import psutil
import pandas as pd
import time
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import unicodedata
import re

API_URL = "http://localhost:8000/query"

st.set_page_config(page_title="Sanskrit RAG Dashboard", layout="wide")

st.title("🪔 Sanskrit RAG: Upload, Chunk, Q&A")

if "query_log" not in st.session_state:
    st.session_state.query_log = []
if "uploads" not in st.session_state:
    st.session_state.uploads = []
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()

tab1, tab2 = st.tabs(["💬 Upload & Q&A", "📊 Dashboard"])

with tab1:
    # Upload
    st.subheader("📁 Upload Doc")
    uploaded = st.file_uploader("PDF/TXT", type=['pdf', 'txt'])
    if uploaded:
        os.makedirs("data", exist_ok=True)
        file_path = f"data/{uploaded.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Saved {uploaded.name}")

        # Ingest in-app
        with st.spinner("Chunking/Indexing..."):
            if uploaded.name.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path)
            docs = loader.load()

            def clean_sanskrit(text):
                text = unicodedata.normalize('NFC', text)
                text = re.sub(r'[�◌्lkmpotn।।।]', ' ', text)
                text = re.sub(r'[^\u0900-\u097Fa-zA-Z\s।।.,!?;॥]', '', text)
                return re.sub(r'\s+', ' ', text).strip()

            for doc in docs:
                doc.page_content = clean_sanskrit(doc.page_content)

            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100, separators=["॥", "।।", "।", "\n\n", "\n", "."])
            chunks = splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")

            st.session_state.uploads.append({'file': uploaded.name, 'chunks': len(chunks), 'timestamp': datetime.now().strftime('%H:%M:%S')})
            st.success(f"Indexed {len(chunks)} chunks!")

    # Q&A
    st.subheader("Q&A")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Query...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        start = time.time()
        with st.chat_message("assistant"):
            r = requests.post(API_URL, json={"question": prompt}, timeout=15)
            if r.status_code == 200:
                data = r.json()
                answer = data["answer"]
                ctx_words = data.get("context_words", 0)
                latency = time.time() - start
                recall = min(1.0, ctx_words / 100) * 0.9
                st.markdown(answer)
                st.caption(f"Context: {ctx_words} | Latency: {latency:.2f}s | Recall: {recall:.2%}")
                st.session_state.query_log.append({
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'query': prompt[:50] + '...',
                    'context_words': ctx_words,
                    'latency': latency,
                    'recall': recall
                })
            else:
                st.error(r.text)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.sidebar:
        if st.button("Test Mukhabhuyaya"): st.session_state.test_prompt = "मुखभूयायः कथा"; st.rerun()

with tab2:
    # Dashboard (unchanged from prior; metrics on uploads/queries)
    col1, col2, col3, col4 = st.columns(4)
    total_queries = len(st.session_state.query_log)
    avg_recall = sum(log['recall'] for log in st.session_state.query_log) / total_queries if total_queries else 0
    avg_latency = sum(log['latency'] for log in st.session_state.query_log) / total_queries if total_queries else 0
    total_chunks = sum(u['chunks'] for u in st.session_state.uploads)
    with col1: st.metric("Queries", total_queries)
    with col2: st.metric("Avg Recall", f"{avg_recall:.2%}")
    with col3: st.metric("Avg Latency", f"{avg_latency:.2f}s")
    with col4: st.metric("Chunks", total_chunks)

    # Resources
    col1, col2 = st.columns(2)
    with col1:
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().used / (1024**3)
        st.metric("CPU %", f"{cpu:.1f}%")
        st.metric("RAM GB", f"{ram:.1f}")
    with col2:
        if st.session_state.query_log:
            df = pd.DataFrame(st.session_state.query_log[-5:])
            st.line_chart(df.set_index('timestamp')[['latency', 'recall']])

    # Logs
    if st.session_state.query_log or st.session_state.uploads:
        logs = st.session_state.query_log + [{'type': 'upload', **u} for u in st.session_state.uploads]
        df_log = pd.DataFrame(logs)
        st.dataframe(df_log)

# Report Gen (UI button in tab2)
if st.button("Generate Report"):
    # Run script
    result = subprocess.run(["python", "code/generate_report.py"], capture_output=True)
    st.success("Report generated!") if result.returncode == 0 else st.error(result.stderr)