# app.py

# Entry point for the Streamlit PDF chat app 

import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Chat with PDFs", layout="wide")
st.title("📄 Chat with Your PDFs")

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="You can upload multiple PDF files to chat with."
)

if uploaded_files:
    st.success(f"Uploaded {len(uploaded_files)} file(s):")
    for file in uploaded_files:
        st.write(f"- {file.name}")

    # Extract text from PDFs
    all_texts = []
    for file in uploaded_files:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        all_texts.append(text)

    # Chunk text (simple split by paragraphs, can be improved)
    chunk_size = 500  # characters
    chunks = []
    for doc_text in all_texts:
        for i in range(0, len(doc_text), chunk_size):
            chunk = doc_text[i:i+chunk_size]
            if chunk.strip():
                chunks.append(chunk)

    st.info(f"Total chunks created: {len(chunks)}")

    # Embed chunks
    with st.spinner("Embedding document chunks..."):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(chunks)
    st.success(f"Embedded {len(embeddings)} chunks.")

    # Create FAISS vector store
    with st.spinner("Creating FAISS vector store..."):
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings).astype('float32'))
    st.success(f"FAISS vector store created with {index.ntotal} vectors.")

    # Store chunks for later retrieval
    st.session_state["chunks"] = chunks
    st.session_state["faiss_index"] = index
    st.session_state["embeddings"] = embeddings

    # --- Hugging Face LLM Integration ---
    st.header("Ask a question about your PDFs")
    user_question = st.text_input("Enter your question:")

    if user_question:
        # Retrieve top relevant chunks
        question_embedding = model.encode([user_question]).astype('float32')
        D, I = index.search(question_embedding, k=5)
        retrieved_chunks = [chunks[i] for i in I[0] if i < len(chunks)]
        context = "\n".join(retrieved_chunks)

        st.write("**Retrieved context:**")
        st.code(context[:1000] + ("..." if len(context) > 1000 else ""))

        with st.spinner("Generating answer with Mistral-7B-Instruct-v0.1..."):
            llm = pipeline(
                "text-generation",
                model="mistralai/Mistral-7B-Instruct-v0.1",
                tokenizer="mistralai/Mistral-7B-Instruct-v0.1",
                max_new_tokens=256,
                do_sample=True,
                temperature=0.2,
                top_p=0.95,
                use_auth_token=True  # Uses HF token from env if set
            )
            prompt = f"Context: {context}\n\nQuestion: {user_question}\nAnswer:"
            result = llm(prompt)
            if isinstance(result, list) and "generated_text" in result[0]:
                answer = result[0]["generated_text"][len(prompt):].strip()
            else:
                answer = str(result)
        st.success("**Answer:**\n" + answer)
else:
    st.info("Please upload one or more PDF files to get started.") 