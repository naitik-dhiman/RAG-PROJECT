import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
load_dotenv()

DB_DIR = "Chroma DB"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_NAME = "openai/gpt-oss-120b"
ALL_BOOKS = "All documents"

st.set_page_config(
    page_title="Document Assistant",
    page_icon="📘",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500;600&display=swap');

    :root {
        --ink: #1b2430;
        --muted: #6b7684;
        --line: #e4e8ee;
        --surface: #f6f8fb;
        --accent: #1f4e79;
        --accent-soft: #e8f0f8;
    }

    html, body, [class*="css"], .stMarkdown, .stChatInput textarea {
        font-family: 'IBM Plex Sans', sans-serif;
        color: var(--ink);
    }

    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
    .block-container { padding-top: 2.5rem; max-width: 780px; }

    /* Page title: one orchestrated entrance */
    .app-title {
        font-family: 'IBM Plex Serif', serif;
        font-size: 2rem;
        font-weight: 600;
        letter-spacing: -0.01em;
        margin: 0;
        animation: settle 0.6s ease-out both;
    }
    .app-subtitle {
        color: var(--muted);
        margin: 0.35rem 0 1.5rem 0;
        font-size: 0.98rem;
        animation: settle 0.6s ease-out 0.1s both;
    }
    @keyframes settle {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--line);
    }
    section[data-testid="stSidebar"] h3 {
        font-family: 'IBM Plex Serif', serif;
        font-size: 1.05rem;
        margin-bottom: 0.25rem;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        border: 1px solid var(--accent);
        background: var(--accent);
        color: #fff;
        font-weight: 500;
        transition: background 0.2s ease, transform 0.15s ease, box-shadow 0.2s ease;
    }
    .stButton > button:hover {
        background: #17395a;
        border-color: #17395a;
        color: #fff;
        box-shadow: 0 4px 12px rgba(31, 78, 121, 0.22);
        transform: translateY(-1px);
    }
    .stButton > button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
    .stButton > button[kind="secondary"] {
        background: transparent;
        color: var(--accent);
    }
    .stButton > button[kind="secondary"]:hover { background: var(--accent-soft); color: var(--accent); }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: transparent;
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.75rem;
        animation: settle 0.35s ease-out both;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: var(--surface);
    }

    /* Chat input */
    [data-testid="stChatInput"] { border-radius: 12px; }
    [data-testid="stChatInput"]:focus-within {
        box-shadow: 0 0 0 2px var(--accent-soft);
        border-color: var(--accent);
    }

    /* Source passages */
    .source-ref {
        font-size: 0.82rem;
        color: var(--muted);
        border-left: 2px solid var(--accent);
        padding-left: 0.7rem;
        margin: 0.5rem 0;
    }
    .status-pill {
        display: inline-block;
        font-size: 0.8rem;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--accent);
        font-weight: 500;
    }
    .empty-state {
        text-align: center;
        color: var(--muted);
        padding: 3rem 1rem;
        border: 1px dashed var(--line);
        border-radius: 12px;
    }

    @media (prefers-reduced-motion: reduce) {
        * { animation: none !important; transition: none !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Cached resources
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def get_vector_store():
    return Chroma(persist_directory=DB_DIR, embedding_function=get_embeddings())


@st.cache_resource(show_spinner=False)
def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY was not found. Add it to your .env file and restart the app.")
        st.stop()
    return init_chat_model(LLM_NAME, model_provider="groq", api_key=api_key)


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant. "
            "Use ONLY the provided context to answer the question. "
            "If the answer is not present in the context, "
            "say: I could not find the answer in the document.",
        ),
        ("human", "Context: {context}\n\nQuestion: {question}"),
    ]
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def list_documents(store: Chroma) -> dict:
    """Return {display_name: source_value} for every indexed document."""
    data = store.get(include=["metadatas"])
    sources = {m.get("source") for m in data["metadatas"] if m and m.get("source")}
    return {Path(s).name: s for s in sorted(sources)}


def index_pdf(uploaded_file, store: Chroma, progress) -> int:
    """Read, chunk, embed and store an uploaded PDF. Returns the chunk count."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        reader = PdfReader(tmp_path)
        docs = [
            Document(
                page_content=page.extract_text() or "",
                metadata={"source": uploaded_file.name, "page": number},
            )
            for number, page in enumerate(reader.pages)
        ]
    finally:
        os.remove(tmp_path)

    docs = [d for d in docs if d.page_content.strip()]
    if not docs:
        raise ValueError("No readable text was found. Scanned PDFs need OCR before upload.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Replace any previous version of the same file
    store._collection.delete(where={"source": uploaded_file.name})

    batch = 64
    for i in range(0, len(chunks), batch):
        store.add_documents(chunks[i : i + batch])
        progress.progress(min((i + batch) / len(chunks), 1.0), text="Indexing document...")
    return len(chunks)


def delete_document(store: Chroma, source: str):
    store._collection.delete(where={"source": source})


def stream_answer(llm, question: str, context: str):
    messages = PROMPT.invoke({"context": context, "question": question})
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content


# ----------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

store = get_vector_store()
documents = list_documents(store)

# ----------------------------------------------------------------------------
# Sidebar: library management
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Library")
    st.caption("Upload a PDF to add it to the searchable knowledge base.")

    uploaded = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded and st.button("Add to library", use_container_width=True):
        progress = st.progress(0.0, text="Reading document...")
        try:
            count = index_pdf(uploaded, store, progress)
            progress.empty()
            st.success(f"Indexed {uploaded.name} ({count} passages).")
            st.rerun()
        except Exception as exc:
            progress.empty()
            st.error(f"Could not index this file. {exc}")

    st.divider()
    st.markdown("### Search scope")

    if documents:
        scope = st.selectbox(
            "Search in",
            [ALL_BOOKS, *documents.keys()],
            label_visibility="collapsed",
        )
        st.markdown(
            f'<span class="status-pill">{len(documents)} document'
            f'{"s" if len(documents) != 1 else ""} indexed</span>',
            unsafe_allow_html=True,
        )

        with st.expander("Manage documents"):
            to_remove = st.selectbox("Remove a document", list(documents.keys()))
            if st.button("Remove from library", type="secondary", use_container_width=True):
                delete_document(store, documents[to_remove])
                st.rerun()
    else:
        scope = ALL_BOOKS
        st.caption("No documents indexed yet.")

    st.divider()
    top_k = st.slider("Passages retrieved", 2, 8, 4)
    show_sources = st.toggle("Show source passages", value=True)

    if st.session_state.messages and st.button("Clear conversation", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ----------------------------------------------------------------------------
# Main: conversation
# ----------------------------------------------------------------------------
st.markdown('<h1 class="app-title">Document Assistant</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-subtitle">Ask questions about your documents. '
    "Answers are drawn only from the uploaded material.</p>",
    unsafe_allow_html=True,
)

if not documents:
    st.markdown(
        '<div class="empty-state">Your library is empty.<br>'
        "Upload a PDF from the sidebar to get started.</div>",
        unsafe_allow_html=True,
    )
    st.stop()


def render_sources(sources):
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(
                f'<div class="source-ref"><strong>{Path(s["source"]).name}, page {s["page"]}</strong><br>'
                f'{s["text"]}</div>',
                unsafe_allow_html=True,
            )


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources") and show_sources:
            render_sources(message["sources"])

query = st.chat_input("Ask a question about your documents")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    search_kwargs = {"k": top_k, "fetch_k": max(top_k * 3, 10), "lambda_mult": 0.5}
    if scope != ALL_BOOKS:
        search_kwargs["filter"] = {"source": documents[scope]}

    retriever = store.as_retriever(search_type="mmr", search_kwargs=search_kwargs)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            docs = retriever.invoke(query)
        context = "\n\n".join(d.page_content for d in docs)

        try:
            answer = st.write_stream(stream_answer(get_llm(), query, context))
        except Exception as exc:
            answer = "The request could not be completed. Please check your connection and API key, then try again."
            st.error(f"{answer} ({exc})")

        sources = [
            {
                "source": d.metadata.get("source", "Unknown"),
                "page": int(d.metadata.get("page", 0)) + 1,
                "text": d.page_content[:350].replace("\n", " ") + "...",
            }
            for d in docs
        ]
        if show_sources and sources:
            render_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})