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
