# 🏥 Medical Q&A Chatbot using RAG (LangChain + HuggingFace)

## 📌 Overview

This project builds a **Medical Question-Answering Chatbot** using the *Gale Encyclopedia of Medicine* PDF dataset.

It uses a **Retrieval-Augmented Generation (RAG)** approach, where relevant information is first retrieved from documents and then passed to a Large Language Model (LLM) to generate accurate answers.

The system ensures responses are **context-based, reliable, and grounded in medical knowledge**, avoiding hallucinations.

---

## 🎯 Objective

The objective of this project is to create a domain-specific chatbot that can answer medical queries using trusted document sources.

It aims to:

* Process and understand large medical PDFs
* Convert text into vector embeddings
* Perform semantic search using FAISS
* Generate accurate answers using an LLM
* Reduce hallucination by restricting answers to context

---

## 📂 Dataset

* Source: **Gale Encyclopedia of Medicine (PDFs)**
* Format: Multiple PDF files stored in `/data` directory

---

## 🛠️ Technologies Used

* Python 🐍
* LangChain 🦜🔗
* HuggingFace 🤗
* FAISS (Vector Database)
* Sentence Transformers
* dotenv (API key management)

---

## ⚙️ Project Workflow

### 1️⃣ Load PDF Documents

* Used `PyPDFDirectoryLoader` to load multiple PDF files

```python
from langchain_community.document_loaders import PyPDFDirectoryLoader
```

---

### 2️⃣ Text Chunking

* Split large text into smaller chunks for better retrieval
* Used `RecursiveCharacterTextSplitter`

```python
chunk_size = 500
chunk_overlap = 50
```

---

### 3️⃣ Generate Embeddings

* Converted text chunks into vector embeddings
* Model used: `all-MiniLM-L6-v2`

```python
HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

---

### 4️⃣ Store in Vector Database (FAISS)

* Stored embeddings locally using FAISS
* Enables fast similarity search

```python
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local("vectorstore/db_faiss")
```

---

### 5️⃣ Load LLM (Mistral)

* Used HuggingFace Inference API
* Model: `Mistral-7B-Instruct`

```python
HuggingFaceEndpoint(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    temperature=0.5
)
```

---

### 6️⃣ Retrieval-Augmented QA Chain

* Combined retriever + LLM using LangChain
* Retrieves top 3 relevant chunks before answering

```python
RetrievalQA.from_chain_type(...)
```

---

### 7️⃣ Custom Prompt Engineering

* Ensures model only answers from context
* Prevents hallucination

```text
"If you dont know the answer, just say that you dont know"
```

---

## 🔄 How It Works

1. User enters a medical query
2. FAISS retrieves relevant document chunks
3. Context is passed to the LLM
4. LLM generates an answer based only on retrieved data

---

## ▶️ How to Run

### 1. Install Dependencies

```bash
pip install langchain langchain-community langchain-huggingface faiss-cpu python-dotenv
```

### 2. Set Environment Variable

Create a `.env` file:

```env
HF_TOKEN=your_huggingface_api_key
```

### 3. Run the Script

```bash
python app.py
```

### 4. Ask Questions

```text
Write Query Here: What is diabetes?
```

---

## 📊 Key Features

* 📚 PDF-based knowledge system
* 🔍 Semantic search using FAISS
* 🤖 LLM-powered responses
* 🛑 Reduced hallucination via prompt control
* ⚡ Fast and efficient retrieval

---

## 🧠 Learning Outcomes

* Understanding of **RAG architecture**
* Working with **LangChain pipelines**
* Vector databases and embeddings
* Prompt engineering techniques
* Building domain-specific AI applications

---

## 🚀 Future Improvements

* Add Streamlit UI for better interaction
* Use advanced embedding models
* Add medical entity recognition
* Deploy on cloud (AWS / HuggingFace Spaces)
* Improve response accuracy with hybrid search

---

## ⚠️ Disclaimer

This chatbot is for **educational purposes only** and should not be used as a substitute for professional medical advice.

---

## ⭐ Conclusion

This project demonstrates how modern AI techniques like **RAG + LLMs** can be used to build intelligent, domain-specific chatbots using real-world data.

---

⭐ *Feel free to fork, star ⭐, and contribute!*
