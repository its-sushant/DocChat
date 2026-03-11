import streamlit as st
from dataclasses import dataclass
from utils import process_uploaded_files, chat

MESSAGES = "messages"
QUERY_ENGINE = "query_engine"
USER = "user"
ASSISTANT = "ai"

FILE_ICONS = {"pdf": "📕", "docx": "📝", "xlsx": "📊", "xls": "📊", "csv": "📋"}

@dataclass
class Message:
    actor: str
    payload: str


st.set_page_config(page_title="DocChat", page_icon="📄", layout="wide")
st.title("DocChat")
st.caption("Chat with your PDF, Word, Excel, and CSV documents powered by GPT-4o-mini")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload Documents")
    st.markdown("Supported: **PDF · DOCX · XLSX · CSV**")

    uploaded_files = st.file_uploader(
        "Choose one or more files",
        type=["pdf", "docx", "xlsx", "xls", "csv"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.subheader("Selected files")
        for f in uploaded_files:
            ext = f.name.rsplit(".", 1)[-1].lower()
            icon = FILE_ICONS.get(ext, "📄")
            st.write(f"{icon} {f.name}")

        if st.button("Process Documents", type="primary", use_container_width=True):
            with st.spinner("Processing… this may take a minute on first run."):
                try:
                    engine = process_uploaded_files(uploaded_files)
                    st.session_state[QUERY_ENGINE] = engine
                    count = len(uploaded_files)
                    st.session_state[MESSAGES] = [
                        Message(
                            actor=ASSISTANT,
                            payload=(
                                f"Ready! I've processed **{count}** file(s). "
                                "Ask me anything about the documents."
                            ),
                        )
                    ]
                    st.success("Documents processed successfully!")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Error: {exc}")

    st.divider()
    st.markdown(
        "**How it works**\n"
        "1. Upload one or more documents\n"
        "2. Click **Process Documents**\n"
        "3. Start chatting!"
    )

# ── Session-state defaults ─────────────────────────────────────────────────────
if MESSAGES not in st.session_state:
    st.session_state[MESSAGES] = [
        Message(
            actor=ASSISTANT,
            payload="👋 Welcome to DocChat! Upload documents in the sidebar to get started.",
        )
    ]

# ── Chat history ───────────────────────────────────────────────────────────────
for msg in st.session_state[MESSAGES]:
    st.chat_message(msg.actor).write(msg.payload)

# ── Chat input ─────────────────────────────────────────────────────────────────
engine_ready = QUERY_ENGINE in st.session_state

if engine_ready:
    prompt = st.chat_input("Ask something about your documents…")
    if prompt:
        st.session_state[MESSAGES].append(Message(actor=USER, payload=prompt))
        st.chat_message(USER).write(prompt)
        with st.spinner("Thinking…"):
            response = chat(prompt, st.session_state[QUERY_ENGINE])
        st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=response))
        st.chat_message(ASSISTANT).write(response)
else:
    st.chat_input("Upload and process documents first…", disabled=True)
