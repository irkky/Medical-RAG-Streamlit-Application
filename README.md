# 🩺 Medical RAG Chatbot

An AI-powered medical assistant that uses Retrieval-Augmented Generation (RAG) to answer questions based on your provided PDF documents.

Built with **Streamlit**, **LangChain**, **Pinecone**, and supports both **Llama 3.1** and **Google Gemini** models.

## 🚀 Prerequisites

Before you begin, ensure you have:
1.  **Python 3.10+** installed.
2.  A **Pinecone** API Key and Index.
3.  A **Hugging Face** Token (for Llama model).
4.  A **Google API Key** (for Gemini model).

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/irkky/Medical-RAG-Streamlit-Application.git
    cd Medical-RAG-Streamlit-Application
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Rename `.env.example` to `.env` and fill in your keys:
    ```bash
    PINECONE_API_KEY=your_key
    PINECONE_INDEX_NAME=your_index
    HUGGINGFACEHUB_API_TOKEN=your_token
    GOOGLE_API_KEY=your_key
    ```

## 📚 Usage

### 1. Ingest Data
Place your medical PDF documents in the `data/raw/` folder. then run:
```bash
python src/ingestion.py
```
This will load, split, embed, and store your documents in Pinecone.

### 2. Run the Chatbot
Start the Streamlit application:
```bash
streamlit run app.py
```

## 🧠 Models
You can switch between models in the sidebar:
- **Llama 3.1**: Runs via Hugging Face Inference API.
- **Gemini 1.5 Flash**: Runs via Google GenAI.

## ⚠️ Disclaimer
This is an AI assistant. **Do not use this for real medical diagnosis or treatment.** Always consult a professional.
