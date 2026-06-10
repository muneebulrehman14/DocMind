"""
DocMind — Local AI Document Chatbot
FLAN-T5-small for fast Q&A over your documents.
No API keys, no cloud calls, 100% offline after first run.
"""

import os, glob, time, hashlib, base64, json
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from typing import List, Tuple

import streamlit as st
import PyPDF2, docx
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

PERSIST_DIR     = "chroma_db"
SAMPLE_DIR      = "sample_docs"
UPLOAD_DIR      = "uploaded_docs"
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 150
COLLECTION_NAME = "my_docs"
MAX_HISTORY     = 5
TOP_K           = 3
LLM_MAX_INPUT   = 512
LLM_MAX_OUTPUT  = 200


def load_icon_b64():
    p = os.path.join(os.path.dirname(__file__), "image.png")
    if os.path.exists(p):
        with open(p, "rb") as f: return base64.b64encode(f.read()).decode()
    return None

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background: #F7F5F0 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: #1a1a1a !important;
    }
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stDecoration"] { display: none; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #FFFFFF !important;
        border-right: 1px solid #E8E4DC !important;
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.8rem; }

    /* ── Brand ── */
    .brand-wrap {
        display: flex; align-items: center; gap: 10px; margin-bottom: 2px;
    }
    .brand-icon {
        width: 38px; height: 38px; border-radius: 10px;
        background: #0D9488;
        display: flex; align-items: center; justify-content: center;
        font-size: 20px; flex-shrink: 0;
    }
    .brand-name {
        font-size: 1.15rem; font-weight: 700; color: #111827; letter-spacing: -0.02em;
    }
    .brand-tag {
        font-size: 0.7rem; color: #9CA3AF; margin-bottom: 1.4rem;
        letter-spacing: 0.02em;
    }

    /* ── Section label ── */
    .sec-label {
        font-size: 0.68rem; font-weight: 600; letter-spacing: 0.09em;
        text-transform: uppercase; color: #9CA3AF; margin-bottom: 6px; margin-top: 4px;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        background: #F9F8F5 !important;
        border: 1.5px dashed #D1C9BC !important;
        border-radius: 10px !important;
        transition: border-color 0.2s;
    }
    [data-testid="stFileUploader"]:hover { border-color: #0D9488 !important; }

    /* ── Buttons ── */
    .stButton > button {
        background: #F9F8F5 !important; color: #374151 !important;
        border: 1px solid #E0DAD0 !important; border-radius: 8px !important;
        font-size: 0.79rem !important; font-weight: 500 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        transition: all 0.15s !important;
    }
    .stButton > button:hover {
        background: #0D9488 !important; color: #fff !important;
        border-color: #0D9488 !important;
    }

    /* ── Main bg ── */
    .block-container { padding-top: 2rem !important; }

    /* ── Page header ── */
    .page-header { margin-bottom: 1.6rem; }
    .page-title {
        font-size: 1.75rem; font-weight: 700; color: #111827;
        letter-spacing: -0.03em; margin: 0 0 4px 0;
    }
    .page-title em {
        font-style: normal; color: #0D9488;
    }
    .page-sub { font-size: 0.83rem; color: #6B7280; }

    /* ── Stats pill ── */
    .stats-pill {
        display: inline-flex; align-items: center; gap: 7px;
        background: #fff; border: 1px solid #E8E4DC;
        border-radius: 20px; padding: 5px 14px;
        font-size: 0.77rem; color: #6B7280; margin-bottom: 1.4rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .live-dot {
        width: 7px; height: 7px; border-radius: 50%; background: #10B981;
        animation: blink 2s ease-in-out infinite;
    }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.35} }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0.4rem 0 !important;
    }
    [data-testid="stChatMessage"] .stMarkdown p {
        color: #374151 !important;
        font-size: 0.91rem !important;
        line-height: 1.75 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    [data-testid="stChatMessageAvatarUser"] {
        background: #F59E0B !important; border-radius: 9px !important;
    }
    [data-testid="stChatMessageAvatarAssistant"] {
        background: #0D9488 !important; border-radius: 9px !important;
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] {
        background: #fff !important;
        border-top: 1px solid #E8E4DC !important;
    }
    [data-testid="stChatInput"] textarea {
        background: #F9F8F5 !important;
        border: 1.5px solid #E0DAD0 !important;
        border-radius: 10px !important;
        color: #111827 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.88rem !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #0D9488 !important;
        box-shadow: 0 0 0 3px rgba(13,148,136,0.1) !important;
    }
    [data-testid="stChatInput"] textarea::placeholder { color: #9CA3AF !important; }

    /* ── Source expander ── */
    [data-testid="stExpander"] {
        background: #fff !important;
        border: 1px solid #E8E4DC !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    }
    [data-testid="stExpander"] summary {
        color: #0D9488 !important; font-size: 0.81rem !important; font-weight: 600 !important;
    }

    /* ── Divider ── */
    hr { border-color: #EDE9E1 !important; }

    /* ── Alert / info ── */
    [data-testid="stAlert"] {
        background: #fff !important; border-radius: 10px !important;
        border: 1px solid #E8E4DC !important; color: #6B7280 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    }

    /* ── Progress bar ── */
    [data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, #0D9488, #14B8A6) !important;
        border-radius: 99px !important;
    }

    /* ── Status widget ── */
    [data-testid="stStatusWidget"] {
        background: #fff !important;
        border: 1px solid #E8E4DC !important;
        border-radius: 10px !important;
    }

    /* ── File chips ── */
    .file-chip {
        display: flex; align-items: center; gap: 7px;
        background: #F9F8F5; border: 1px solid #E8E4DC;
        border-radius: 7px; padding: 5px 10px;
        font-size: 0.74rem; color: #374151; margin: 3px 0;
    }
    .file-chip-icon { color: #0D9488; font-size: 0.85rem; }

    /* ── Empty state ── */
    .empty-wrap {
        text-align: center; padding: 3.5rem 1rem;
    }
    .empty-icon { font-size: 2.8rem; margin-bottom: 0.8rem; display: block; }
    .empty-title { font-size: 1rem; font-weight: 600; color: #374151; margin-bottom: 5px; }
    .empty-body { font-size: 0.82rem; color: #9CA3AF; line-height: 1.6; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #D1C9BC; border-radius: 99px; }

    /* ── Misc ── */
    .stCaption { color: #9CA3AF !important; font-size: 0.76rem !important; }
    h3 {
        color: #9CA3AF !important; font-size: 0.72rem !important;
        font-weight: 600 !important; text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }
    </style>
    """, unsafe_allow_html=True)


def init_state():
    for k, v in {"messages":[],"collection":None,"chroma":None,"processed":set(),"llm":None,"embedder":None}.items():
        st.session_state.setdefault(k, v)


def parse_pdf(path):
    parts = []
    with open(path, "rb") as fh:
        for page in PyPDF2.PdfReader(fh).pages:
            t = page.extract_text()
            if t: parts.append(t)
    return "\n".join(parts)

def parse_docx(path):
    return "\n".join(p.text for p in docx.Document(path).paragraphs if p.text.strip())

def parse_txt(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()

def parse_file(path):
    ext = os.path.splitext(path)[1].lower()
    handlers = {".pdf": parse_pdf, ".docx": parse_docx, ".txt": parse_txt}
    if ext not in handlers: raise ValueError(f"Unsupported: {ext}")
    return handlers[ext](path)


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", "! ", "? "]:
                idx = text.rfind(sep, start, end)
                if idx != -1 and idx - start > size // 3:
                    end = idx + len(sep); break
        chunk = text[start:end].strip()
        if chunk: chunks.append(chunk)
        next_start = end - overlap
        if next_start <= start: break
        start = next_start
    return chunks


@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedder(): return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Loading LLM (FLAN-T5-small)…")
def load_llm():
    mid = "google/flan-t5-small"
    tok = AutoTokenizer.from_pretrained(mid)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(mid)
    mdl.eval()
    return mdl, tok

@st.cache_resource(show_spinner="Initialising vector store…")
def init_chroma():
    os.makedirs(PERSIST_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=PERSIST_DIR)

def get_or_create_coll(client):
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        return client.get_collection(COLLECTION_NAME, embedding_function=ef)
    return client.create_collection(COLLECTION_NAME, embedding_function=ef)

def _chunk_id(source, idx, text):
    return f"{source}_{idx}_{hashlib.md5(text.encode()).hexdigest()[:8]}"

def add_chunks(coll, chunks, source, embedder, status_el):
    if not chunks: return 0
    try:
        ex = coll.get(where={"source": source})
        if ex["ids"]: coll.delete(ids=ex["ids"])
    except Exception: pass
    batch_size, all_embeds, total = 16, [], len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i:i+batch_size]
        status_el.info(f"Embedding {i+1}–{min(i+batch_size,total)} of {total} chunks…")
        all_embeds.extend(embedder.encode(batch, show_progress_bar=False).tolist())
    coll.add(
        documents=chunks, embeddings=all_embeds,
        metadatas=[{"source": source, "idx": i} for i in range(total)],
        ids=[_chunk_id(source, i, c) for i, c in enumerate(chunks)],
    )
    return total

def search_docs(coll, query, k=TOP_K):
    k = min(k, coll.count())
    if k == 0: return []
    res = coll.query(query_texts=[query], n_results=k, include=["documents","metadatas","distances"])
    out = []
    if res["documents"] and res["documents"][0]:
        for i, doc in enumerate(res["documents"][0]):
            src  = res["metadatas"][0][i]["source"]
            dist = res["distances"][0][i] if res["distances"] else 0.0
            out.append((doc, src, dist))
    return out


def make_prompt(query, context, history):
    parts = []
    if history:
        parts.append("Conversation history:")
        for q, a in history[-3:]: parts.append(f"Q: {q}\nA: {a}")
        parts.append("")
    if context: parts.append(f"Context:\n{context[:2000]}")
    parts.append(f"Question: {query}")
    parts.append("Answer using ONLY the context. If not found, say: I don't know.")
    return "\n".join(parts)

def generate_answer(llm, query, context, history):
    model, tokenizer = llm
    prompt = make_prompt(query, context, history)
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=LLM_MAX_INPUT)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=LLM_MAX_OUTPUT, do_sample=False, num_beams=4, early_stopping=True)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return answer or "I couldn't find a relevant answer in the documents."
    except Exception as exc:
        return f"Error: {exc}"


def typewrite(text, placeholder, delay=0.004):
    out = ""
    for ch in text:
        out += ch
        placeholder.markdown(out + "▌")
        time.sleep(delay)
    placeholder.markdown(out)


def seed_samples():
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    if glob.glob(os.path.join(SAMPLE_DIR, "*")): return
    with open(os.path.join(SAMPLE_DIR, "ai_overview.txt"), "w", encoding="utf-8") as f:
        f.write("Artificial Intelligence (AI) is a branch of computer science founded in 1956 at Dartmouth College.\n\nMachine Learning gives computers the ability to learn without explicit programming. Three paradigms: supervised, unsupervised, and reinforcement learning.\n\nDeep Learning uses multi-layer neural networks for breakthroughs in image recognition, speech processing and NLP.\n\nNLP enables computers to read, understand and generate human language — powering translation, sentiment analysis and chatbots.\n\nAI is transforming healthcare, finance, transportation, entertainment and education.\n\nFuture directions include quantum ML, neuromorphic chips and AGI — safety and alignment remain critical.")
    try:
        d = docx.Document()
        d.add_heading("Python Programming Language", 0)
        d.add_paragraph("Python is a high-level language created by Guido van Rossum in 1991.")
        d.add_heading("Key Features", 1)
        d.add_paragraph("Python supports procedural, OOP and functional paradigms with a vast ecosystem.")
        d.add_heading("Data Science & ML", 1)
        d.add_paragraph("NumPy, pandas, scikit-learn, TensorFlow and PyTorch make Python dominant in data science.")
        d.save(os.path.join(SAMPLE_DIR, "python_info.docx"))
    except Exception: pass

def auto_index_samples(coll, embedder):
    if coll.count() > 0: return
    files = glob.glob(os.path.join(SAMPLE_DIR, "*"))
    if not files: return
    ph = st.empty(); ph.info("Auto-indexing sample documents…")
    for path in files:
        name = os.path.basename(path)
        if name in st.session_state.processed: continue
        try:
            text = parse_file(path)
            if text.strip():
                chunks = chunk_text(text)
                add_chunks(coll, chunks, name, embedder, ph)
                st.session_state.processed.add(name)
        except Exception: pass
    ph.empty()

def render_sources(sources):
    if not sources: return
    st.markdown("---")
    st.markdown("**📎 Sources**")
    for i, (doc, src, score) in enumerate(sources):
        st.markdown(
            f'<div style="font-size:0.8rem;color:#0D9488;font-weight:600;">'
            f'#{i+1} · {src} <span style="color:#9CA3AF;font-weight:400;">(relevance {score:.3f})</span></div>',
            unsafe_allow_html=True
        )
        st.markdown(doc)
        if i < len(sources) - 1: st.divider()

TRACKER_FILE = ".indexed_files.json"

def _load_tracker():
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, "r") as f: return set(json.load(f))
    return set()

def _save_tracker(ids):
    with open(TRACKER_FILE, "w") as f: json.dump(sorted(ids), f)

def process_upload(f, coll, embedder, prog, msg, i, total):
    msg.info(f"Processing **{f.name}** ({i+1}/{total})…")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(UPLOAD_DIR, f.name)
    with open(dest, "wb") as buf: buf.write(f.getbuffer())
    try:
        text = parse_file(dest)
        if text.strip():
            chunks = chunk_text(text)
            n = add_chunks(coll, chunks, f.name, embedder, msg)
            st.session_state.processed.add(f.name)
            tracked = _load_tracker(); tracked.add(f.name); _save_tracker(tracked)
            msg.success(f"✅ **{f.name}** — {n} chunks indexed")
        else:
            msg.warning(f"⚠️ **{f.name}** — no readable text found")
    except Exception as exc:
        msg.error(f"❌ **{f.name}** — {exc}")

def reindex_uploads(coll, embedder):
    tracked = _load_tracker()
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    files = [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]
    new_files = [f for f in files if f not in tracked]
    if not new_files: return
    total = len(new_files)
    prog = st.progress(0, text=f"Re-indexing {total} saved file(s)…")
    for i, name in enumerate(new_files):
        path = os.path.join(UPLOAD_DIR, name)
        try:
            text = parse_file(path)
            if text.strip():
                chunks = chunk_text(text)
                add_chunks(coll, chunks, name, embedder, prog)
                st.session_state.processed.add(name)
                tracked.add(name)
        except Exception:
            pass
        prog.progress((i + 1) / total)
    _save_tracker(tracked)
    prog.empty()


def main():
    st.set_page_config(page_title="DocMind — Local AI", page_icon="image.png", layout="wide", initial_sidebar_state="expanded")
    inject_css()
    init_state()
    seed_samples()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        b64 = load_icon_b64()
        if b64:
            st.markdown(f'<div style="text-align:center;margin:0 0 6px 0;"><img src="data:image/png;base64,{b64}" style="width:80px;height:80px;border-radius:16px;"></div>', unsafe_allow_html=True)

        if st.session_state.llm is None:
            with st.status("Starting up…", expanded=True) as status:
                st.write("📦 Embedding model…")
                embedder = load_embedder()
                st.write("✅ Embedding model ready")
                st.write("📦 Vector database…")
                chroma = init_chroma()
                coll   = get_or_create_coll(chroma)
                st.write(f"✅ Vector DB ready ({coll.count()} chunks)")
                st.write("📦 FLAN-T5-small LLM…")
                llm = load_llm()
                st.write("✅ LLM ready")
                st.session_state.update({"llm":llm,"embedder":embedder,"chroma":chroma,"collection":coll})
                status.update(label="✅ All systems ready", state="complete")
            auto_index_samples(st.session_state.collection, st.session_state.embedder)
            reindex_uploads(st.session_state.collection, st.session_state.embedder)

        st.divider()
        st.markdown('<div class="sec-label">Upload documents</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "PDF · DOCX · TXT", type=["pdf","docx","txt"],
            accept_multiple_files=True, label_visibility="collapsed"
        )

        if uploaded:
            coll = st.session_state.collection
            embedder = st.session_state.embedder
            if coll is None or embedder is None:
                st.warning("Models still loading — please wait.")
            else:
                new_files = [f for f in uploaded if f.name not in st.session_state.processed]
                if new_files:
                    prog = st.progress(0); msg = st.empty()
                    for i, f in enumerate(new_files):
                        process_upload(f, coll, embedder, prog, msg, i, len(new_files))
                    time.sleep(0.6); msg.empty(); prog.empty()
                    st.rerun()

        if st.session_state.processed:
            st.divider()
            st.markdown('<div class="sec-label">Indexed files</div>', unsafe_allow_html=True)
            for fn in sorted(st.session_state.processed):
                ext  = fn.rsplit(".", 1)[-1].upper() if "." in fn else "?"
                icon = {"PDF":"📄","DOCX":"📝","TXT":"📃"}.get(ext,"📁")
                st.markdown(f'<div class="file-chip"><span class="file-chip-icon">{icon}</span>{fn}</div>', unsafe_allow_html=True)

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑 Clear chat", use_container_width=True):
                st.session_state.messages = []; st.rerun()
        with c2:
            if st.button("🔄 Reset all", use_container_width=True):
                st.session_state.messages = []; st.session_state.processed = set()
                try: st.session_state.chroma.delete_collection(COLLECTION_NAME)
                except Exception: pass
                st.session_state.collection = get_or_create_coll(st.session_state.chroma)
                st.rerun()

    # ── Main ─────────────────────────────────────────────────────────────────
    coll = st.session_state.collection

    st.markdown("""
    <div class="page-header">
        <div class="page-title">Chat with your <em>documents</em></div>
        <div class="page-sub">Answers come from your files — private, local, no internet needed</div>
    </div>
    """, unsafe_allow_html=True)

    if coll is None:
        st.info("⏳ Loading… please wait."); return

    count = coll.count()
    if count > 0:
        n = len(st.session_state.processed)
        st.markdown(
            f'<div class="stats-pill"><div class="live-dot"></div>'
            f'{count} chunks indexed · {n} file{"s" if n!=1 else ""}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown("""
        <div class="empty-wrap">
            <span class="empty-icon">📂</span>
            <div class="empty-title">No documents loaded</div>
            <div class="empty-body">Upload PDF, DOCX, or TXT files using the sidebar.<br>Sample documents load automatically on first run.</div>
        </div>""", unsafe_allow_html=True)
        if glob.glob(os.path.join(SAMPLE_DIR, "*")) and st.session_state.embedder:
            if st.button("📥 Load sample documents"):
                auto_index_samples(coll, st.session_state.embedder); st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"): render_sources(msg["sources"])

    if prompt := st.chat_input("Ask anything about your documents…"):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner("Searching…"):
                docs = search_docs(coll, prompt) if count > 0 else []

            if not docs:
                answer  = "I couldn't find relevant information in the loaded documents. Try uploading more files or rephrasing your question."
                sources = None
                typewrite(answer, placeholder)
            else:
                context = "\n\n".join(d[0] for d in docs)
                history: List[Tuple[str, str]] = []
                msgs = st.session_state.messages
                for j in range(0, len(msgs) - 1, 2):
                    if msgs[j]["role"] == "user" and msgs[j+1]["role"] == "assistant":
                        history.append((msgs[j]["content"], msgs[j+1]["content"]))
                history = history[-MAX_HISTORY:]
                with st.spinner("Generating answer…"):
                    answer = generate_answer(st.session_state.llm, prompt, context, history)
                typewrite(answer, placeholder)
                sources = list(docs)
                render_sources(sources)

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        if len(st.session_state.messages) > MAX_HISTORY * 2:
            st.session_state.messages = st.session_state.messages[-(MAX_HISTORY * 2):]


if __name__ == "__main__":
    main()