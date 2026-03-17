# 📡 Agentic RAG System

A premium, high-performance **Retrieval-Augmented Generation (RAG)** web application engineered with **Rust**, **HTMX**, and **Groq**. This system transforms your local documents into an interactive knowledge base with agentic tool-use capabilities.

---

## ✨ Features

- **🚀 Lightning Fast Inference**: Leveraging **Groq API** with Llama 3.3 70B for near-instant responses.
- **🛡️ Private Local Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` via **HuggingFace Candle**.
- **🤖 Agentic Reasoning**: The system doesn't just retrieve; it uses the `search_documents` tool to intelligently query its own knowledge base.
- **🏗️ Industrial Tech Stack**:
  - **Backend**: Axum (high-concurrency web framework).
  - **Database**: SQLite with WAL mode for efficient concurrent read/writes.
  - **Text Extraction**: Support for PDF and TXT (DOCX placeholder).
- **💅 Premium UI/UX**:
  - **HTMX**: For a seamless, SPA-like feel without JavaScript fatigue.
  - **Tailwind CSS**: Sleek, modern design with loading indicators and responsive layouts.
  - **Real-time Feedback**: "Thinking..." indicators and dynamic file status updates.

---

## 🛠️ Prerequisites

- **Rust Toolchain**: [Install Rust](https://rustup.rs/)
- **Groq API Key**: Obtain from [Groq Console](https://console.groq.com/)

---

## 🚀 Quick Start

### 1. Configure Environment

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_key_here
PORT=3000
```

### 2. Launch the Application

```bash
cargo run --release
```

Access the dashboard at `http://127.0.0.1:3000`.

---

## 📖 How It Works

1.  **Ingestion**: Files are uploaded through a multipart form.
2.  **Processing**: The backend extracts text and chunks it using a semantic sliding window strategy.
3.  **Embedding**: Each chunk is transformed into a 384-dimensional vector using `all-MiniLM-L6-v2`.
4.  **Indexing**: Vectors and content are stored in a local SQLite database (`rag.db`).
5.  **Retrieval**: When you ask a question, the Agentic LLM generates a search query, which is cross-referenced against your index using **Cosine Similarity**.
6.  **Synthesis**: The LLM receives the most relevant context and synthesizes a concise, accurate answer.

## 🌍 Deployment

This project is optimized for **Render**'s free tier.

### Why Render?

- **Per-Project Limits**: Each app gets its own **512MB RAM**, which is plenty for this Rust backend.
- **Smart "Sleep"**: Apps spin down after 15 mins of inactivity, preserving your **750 free monthly hours**. This allows you to host dozens of projects on one account!
- **Docker Ready**: The included `Dockerfile` bakes the embedding model into the image, ensuring **instant cold starts** so you don't have to wait for models to download when the server wakes up.

### Deployment Steps

1. Push your code to a GitHub repository.
2. Sign up/Log in to [Render](https://render.com/).
3. Click **New +** > **Web Service**.
4. Connect your repo.
5. Render will automatically detect the `Dockerfile`.
6. Add your `GROQ_API_KEY` in the **Environment Variables** section.
7. Click **Deploy**.

---

## 📄 License

This project is open-source and intended for educational and professional demonstration purposes.
