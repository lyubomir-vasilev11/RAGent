import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
import streamlit as st
from app import main
import os

st.set_page_config(page_title="The King RAGent", page_icon="👑", layout="wide")

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "initialized" not in st.session_state:
        st.session_state.initialized = False

def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def main_chat():
    # Create two columns: sidebar and main content
    with st.sidebar:
        st.title("Settings")
        dry_run = st.checkbox("🔧 Dry Run Mode", value=False, help="Run without making API calls")
        if dry_run:
            st.warning("Test mode active - using stub responses")
    
    # Main content
    st.title("👑 The King RAGent")
    st.markdown("### Your AI Research Assistant")
    
    initialize_session_state()
    
    # PDF Upload
    if not st.session_state.initialized:
        uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
        if uploaded_file is not None:
            # Save the uploaded file temporarily
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Initialize the vector store
            response = main(None, initialize=True, pdf_path="temp.pdf", dry_run=dry_run)
            st.session_state.initialized = True
            os.remove("temp.pdf")  # Clean up
            st.rerun()

    # Display chat history
    display_chat_history()
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = main(prompt, initialize=False, dry_run=dry_run)
                
        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Rerun to update the chat display
        st.rerun()

if __name__ == "__main__":
    main_chat()