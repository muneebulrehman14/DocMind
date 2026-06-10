# 🧠 DocMind — Local AI Document Chatbot

**100% offline · No API keys · Privacy-first**

DocMind lets you chat with your documents (PDF, DOCX, TXT) using a local AI model. Everything runs on your machine — no cloud, no data leaving your computer.

![Demo](https://img.shields.io/badge/status-working-brightgreen)
![Python](https://img.shields.io/badge/python-3.9+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## ✨ Features

- ✅ **100% Local** — No API keys, no internet required after setup
- ✅ **Multi-format** — Upload PDF, DOCX, or TXT files
- ✅ **Smart RAG** — Retrieves relevant chunks from your documents
- ✅ **Conversation Memory** — Remembers last 5 exchanges
- ✅ **Source Citations** — See exactly where answers come from
- ✅ **Streaming Response** — Typewriter effect for natural feel
- ✅ **Persistent Storage** — Documents stay indexed across sessions

## 🏗️ Architecture
┌─────────────┐ ┌──────────────┐ ┌─────────────────┐
│ Streamlit │────▶│ Chroma DB │◀────│ Documents │
│ UI │ │ (Vectors) │ │ (PDF/DOCX/TXT) │
└─────────────┘ └──────────────┘ └─────────────────┘
│ │ │
▼ ▼ ▼
┌─────────────┐ ┌──────────────┐ ┌─────────────────┐
│ FLAN-T5 │◀────│ Sentence- │────▶│ Chunking & │
│ (LLM) │ │ Transformers │ │ Embedding │
└─────────────┘ └──────────────┘ └─────────────────┘

## 📋 Requirements

- **Python 3.9+**
- **8GB RAM** (4GB minimum, slower)
- **3GB free disk space** (for models)

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/muneebulrehman14/DocMind-Local-Chatbot.git
cd DocMind-Local-Chatbot
**2. Create virtual environment**
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

** 3. Install dependencies**
pip install -r requirements.txt

**Run the app
streamlit run app.py**


**🎮 Usage Guide**
Upload Documents

    Click sidebar expander

    Drag & drop PDF/DOCX/TXT files

    Wait for "✅ Indexed" message

Ask Questions

    Type in chat input at bottom

    Bot searches relevant chunks

    Answer appears with source citations

Manage Sessions

    Clear Chat — Removes conversation history

    Reset All — Deletes all documents and starts fresh
