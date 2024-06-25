import streamlit as st
from dataclasses import dataclass
from utils import get_query_engine, chat

@dataclass
class Message:
    actor: str
    payload: str

USER = "user"
ASSISTANT = "ai"
MESSAGES = "messages"
QUERY_ENGINE = "query_engine"

st.title("DocChat (A Friendly ChatBot to talk with pdf documents.)")

if MESSAGES not in st.session_state:
    st.session_state[MESSAGES] = [Message(actor=ASSISTANT, payload="Ask me something...")]
if QUERY_ENGINE not in st.session_state:
    st.session_state[QUERY_ENGINE] = get_query_engine()

msg: Message
for msg in st.session_state[MESSAGES]:
    st.chat_message(msg.actor).write(msg.payload)

prompt: str = st.chat_input("Enter a prompt here")

if prompt:
    st.session_state[MESSAGES].append(Message(actor=USER, payload=prompt))
    st.chat_message(USER).write(prompt)
    response: str = chat(prompt, st.session_state[QUERY_ENGINE])
    st.session_state[MESSAGES].append(Message(actor=ASSISTANT, payload=response))
    st.chat_message(ASSISTANT).write(response)