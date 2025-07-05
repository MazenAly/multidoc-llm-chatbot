# multidoc-llm-chatbot

"""
README.md

📄 Chat with PDFs – Free LLM-Powered Q&A App

This project is a lightweight, interactive PDF question-answering tool built with **LangChain**, **Hugging Face LLMs**, and **Streamlit**. It allows users to upload one or multiple PDF documents and ask questions about their content in natural language.

✨ Features:
- 📁 Upload and process **multiple PDFs**
- 🧠 Uses **Hugging Face embeddings** and **Mistral-7B-Instruct** (or any HF-compatible LLM)
- 🔍 Powered by **vector search** and **Retrieval-Augmented Generation (RAG)**
- 💬 Ask open-ended or specific questions about the uploaded files
- 🌐 Clean and easy-to-use **Streamlit web interface**
- 🔓 100% free to run — no OpenAI API key required

🛠️ Technologies:
- Retrieval-Augmented Generation (RAG)
- FAISS for vector storage
- Hugging Face Hub (LLMs + Embeddings)
- Streamlit for frontend
- PyPDF for PDF loading

🚀 Setup:
1. Clone the repo and install dependencies:
```bash
pip install langchain faiss-cpu unstructured pypdf tiktoken streamlit huggingface_hub
```

2. (Optional) Set your Hugging Face token in an environment variable:
```bash
export HUGGINGFACEHUB_API_TOKEN=your-hf-token
```

3. Run the app:
```bash
streamlit run app.py
```

📦 Note:
Make sure you have access to the model you choose on Hugging Face. The default is `mistralai/Mistral-7B-Instruct-v0.1`, but you can change it.

"""
