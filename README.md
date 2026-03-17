# 📡 Agentic RAG System

A high-performance **Retrieval-Augmented Generation (RAG)** web application engineered with **Rust**, **HTMX**, and **Groq**. This system transforms local documents into an interactive knowledge base with autonomous agentic capabilities.

---

## ✨ Features

- **🚀 Dual-Model Resilience**: Intelligent fallback between Llama 3.3 70B and Llama 4 Scout to circumvent rate limits.
- **🛡️ Private Local Embeddings**: Built-in 384-dimensional vectorization using `all-MiniLM-L6-v2` via **HuggingFace Candle**.
- **🤖 Agentic Reasoning**: An autonomous loop that uses the `search_documents` tool to query the local index before synthesizing answers.
- **🏗️ Industrial Tech Stack**:
  - **Backend**: Axum (high-concurrency asynchronous web framework).
  - **Database**: SQLite with **WAL mode** for high-throughput concurrent access.
  - **Frontend**: HTMX for reactive, SPA-like interactions without the complexity of modern JS frameworks.
- **💅 Noir Aesthetic**: A premium, terminal-inspired dark UI powered by Tailwind CSS.

---

## 🛠️ Infrastructure & Resilience

### API Fallback & Retry Strategy

This system is designed to handle the constraints of the **Groq Free Tier** gracefully:

1. **Primary Model**: Defaults to `llama-3.3-70b-versatile` for high-reasoning tasks.
2. **Momentary Limits (TPM/RPM)**: Automatically parses `retry-after` headers and performs jittered retries (up to 3 times) before failing.
3. **Daily Limits (TPD/RPD)**: Upon detecting a daily quota exhaustion, the system immediately switches to `llama-4-scout-17b-instruct` to maintain service continuity.
4. **Gas Gauge Monitoring**: Real-time logging of `x-ratelimit-remaining` headers provides visibility into API consumption.

---

## 🚀 Getting Started Locally

### 1. Prerequisites

- **Rust Toolchain**: [Install via rustup](https://rustup.rs/) (Ensure you have `cargo` and `rustc` stable).
- **Groq API Key**: Obtain a free-tier key from the [Groq Console](https://console.groq.com/).
- **System Dependencies**:
  - **Windows**: Build tools for Visual Studio (C++ workload) are required for compiling certain dependencies like `rusqlite`.
  - **Linux**: `sudo apt-get install pkg-config libssl-dev build-essential`
  - **macOS**: `brew install openssl pkg-config`
- **Environment**: A `.env` file in the root directory (see below).

### 2. Configure Environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_key_here
PORT=3000
```

### 3. Launch the Application

```bash
cargo run --release
```

Access the dashboard at `http://localhost:3000`.

---

## 📖 Architectural Overview

1.  **Ingestion**: Documents (PDF, TXT, DOCX) are processed and cleaned server-side.
2.  **Chunking**: Text is split using a semantic sliding window to preserve context across boundaries.
3.  **Vectorization**: Chunks are embedded into a vector space using a local BERT model (MiniLM).
4.  **Retrieval**: Queries trigger a semantic search against the SQLite index using Cosine Similarity.
5.  **Synthesis**: The agentic reasoning loop evaluates retrieved context and generates a precise response, citing source material where applicable.

---

## 🌍 Deployment

The system is optimized for containerized environments like **Render**.

### Render Configuration

- **RAM**: 512MB (Free Tier compatible)
- **Startup**: The included `Dockerfile` pre-downloaded the embedding model, ensuring sub-second cold starts.
- **Persistence**: Uses an ephemeral SQLite database; for production, consider mounting a volume or using an external DB provider.

---

## 📄 License

This project is open-source and intended for professional demonstration and educational purposes.
