import streamlit as st
import requests
import uuid
from datetime import datetime

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Document Q&A System",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.title("📚 Document Q&A")
    st.markdown("---")
    
    # Info
    st.markdown("**Knowledge Base:** Pre-loaded with healthcare documents.")
    st.markdown("---")
    
    # Session info
    st.caption(f"Session: {st.session_state.session_id[:8]}...")
    if st.button("New Chat"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    
    # Health check
    try:
        health = requests.get(f"{API_URL}/health", timeout=2)
        if health.status_code == 200:
            st.success("✅ Backend connected")
        else:
            st.error("❌ Backend error")
    except:
        st.error("❌ Backend unreachable")

# Main chat area
st.title("💬 Chat with Your Documents")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📖 Sources"):
                for i, source in enumerate(message["sources"][:3], 1):
                    st.caption(f"Document {i} (score: {source['score']:.2f})")
                    st.info(source['content'][:300] + "...")

# Chat input
if prompt := st.chat_input("Ask a question about your documents"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={
                        "query": prompt,
                        "session_id": st.session_state.session_id
                    },
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                
                st.markdown(data["answer"])
                
                # Show sources
                if data["sources"]:
                    with st.expander("📖 Sources"):
                        for i, source in enumerate(data["sources"][:3], 1):
                            st.caption(f"Document {i} (score: {source['score']:.2f})")
                            st.info(source['content'][:300] + "...")
                
                # Add to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "sources": data["sources"]
                })
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
