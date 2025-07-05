# 📌 Development Plan: Chat with PDFs (Streamlit + Free LLM)

## 🎯 Goal

Develop a lightweight and professional web app that allows users to upload one or more PDF files and ask questions about them using a free, open-source LLM via Hugging Face. The app should run without any OpenAI API key and be easily deployable.

---

## 🗂️ Project Structure

```
chat-with-pdfs/
├── app.py                  # Main Streamlit app
├── requirements.txt        # Project dependencies
├── README.md               # Project overview and usage
├── .env                    # (Optional) HF token for local dev
└── data/                   # Example PDFs for testing
```

---

## 🔨 Features To Implement

### ✅ MVP

* [x] Upload multiple PDF files
* [x] Chunk and embed documents using `sentence-transformers`
* [x] Create FAISS vector store for retrieval
* [x] Use Hugging Face-hosted model (`mistralai/Mistral-7B-Instruct-v0.1`)
* [x] RAG-based QA system via LangChain’s `RetrievalQA`
* [x] Streamlit UI with input and output display

### 🚀 Stretch Features

* [ ] Add support for chat history (multi-turn conversation)
* [ ] Add file type support for `.docx`, `.txt`
* [ ] Model selector (allow switching between models like LLaMA, Falcon, etc.)
* [ ] Cache vectorstore to speed up repeat queries

---

## ⚙️ Tech Stack

* **LangChain** – LLM chaining and document handling
* **FAISS** – Local vector store
* **Streamlit** – Frontend
* **Hugging Face Hub** – Embeddings + LLMs (free tier)
* **PyPDF** – PDF parsing

---

## 📦 Setup Checklist

1. [ ] Create virtual environment
2. [ ] Install dependencies with `pip install -r requirements.txt`
3. [ ] Add `.env` file or export `HUGGINGFACEHUB_API_TOKEN`
4. [ ] Run using `streamlit run app.py`
5. [ ] Upload test PDFs and ask questions

---

## 📈 Milestones

| Week | Milestone                                  |
| ---- | ------------------------------------------ |
| 1    | Core functionality working locally         |
| 2    | Polish UI, add README and requirements.txt |
| 3    | Stretch features + public GitHub release   |

---

## 🧪 Example Prompts

* "What are the main findings in this research paper?"
* "When does the contract expire?"
* "Summarize the executive summary section."

---

