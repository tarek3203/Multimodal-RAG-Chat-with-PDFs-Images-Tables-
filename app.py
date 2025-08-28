import streamlit as st
import os
from dotenv import load_dotenv
from pdf_processor import PDFProcessor
from rag_system import RAGSystem
from typing import List

# Load environment variables
load_dotenv()

def initialize_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()
    if 'pdf_processor' not in st.session_state:
        st.session_state.pdf_processor = PDFProcessor()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []

def process_uploaded_files(uploaded_files: List) -> bool:
    """Process uploaded PDF files and return success status"""
    success_count = 0
    
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.processed_files:
            try:
                # Process PDF
                pdf_bytes = uploaded_file.read()
                processed_doc = st.session_state.pdf_processor.process_pdf(
                    pdf_bytes, uploaded_file.name
                )
                
                # Add to RAG system
                st.session_state.rag_system.add_documents([processed_doc])
                
                # Track processed files
                st.session_state.processed_files.append(uploaded_file.name)
                success_count += 1
                
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
    
    return success_count > 0

def get_ai_response(user_message: str, has_documents: bool) -> str:
    """Get AI response based on whether documents are available"""
    if has_documents:
        # Use RAG system for document-based queries
        return st.session_state.rag_system.query(user_message)
    else:
        # Use direct LLM for normal conversation
        return st.session_state.rag_system.chat_without_documents(user_message)

def main():
    st.set_page_config(
        page_title="AI Chatbot with RAG",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("🤖 AI Chatbot with Document Analysis")
    st.markdown("*Chat normally or upload PDFs for document-based conversations*")
    
    # Sidebar with file management
    with st.sidebar:
        st.header("📁 Document Library")
        
        # Display processed files
        if st.session_state.processed_files:
            st.subheader("📋 Processed Documents")
            for i, filename in enumerate(st.session_state.processed_files, 1):
                st.text(f"{i}. {filename}")
        else:
            st.info("No documents uploaded yet")
        
        # Clear memory button
        if st.button("🗑️ Clear Chat & Documents", type="secondary", use_container_width=True):
            st.session_state.rag_system.clear_memory()
            st.session_state.messages = []
            st.session_state.processed_files = []
            st.success("✅ Everything cleared!")
            st.rerun()
        
        # Instructions
        st.markdown("---")
        st.markdown("### 💡 How to use:")
        st.markdown("1. **Normal chat:** Just type your message")
        st.markdown("2. **With documents:** Upload PDFs using the attachment button in chat")
        st.markdown("3. **Ask questions:** Query your uploaded documents")
    
    # Chat interface
    st.header("💬 Chat")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show file attachments if any
            if "files" in message and message["files"]:
                st.caption(f"📎 Attached: {', '.join(message['files'])}")
    
    # File uploader in expandable section above chat input
    with st.expander("📎 Attach Documents", expanded=False):
        uploaded_files = st.file_uploader(
            "Choose PDF files to chat with",
            type=["pdf"],
            accept_multiple_files=True,
            key="chat_file_uploader",
            help="Upload PDFs to enable document-based conversations"
        )
        
        if uploaded_files:
            if st.button("📤 Upload & Process Files", use_container_width=True):
                with st.spinner("Processing documents..."):
                    if process_uploaded_files(uploaded_files):
                        st.success(f"✅ Processed {len(uploaded_files)} file(s)")
                        st.rerun()
    
    # Chat input
    if prompt := st.chat_input("Type your message here... (Upload documents above for document-based chat)"):
        # Determine if we have documents
        has_documents = len(st.session_state.processed_files) > 0
        
        # Add user message to chat history
        user_message = {
            "role": "user", 
            "content": prompt,
            "files": st.session_state.processed_files.copy() if has_documents else []
        }
        st.session_state.messages.append(user_message)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            if user_message["files"]:
                st.caption(f"📎 Available documents: {', '.join(user_message['files'])}")
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_ai_response(prompt, has_documents)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Auto-scroll to bottom
        st.rerun()
    
    # Footer info
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("🔍 **Vision:** OpenAI GPT-4V for PDF processing")
    with col2:
        st.caption("💬 **Chat:** Groq Mixtral for fast responses")
    with col3:
        st.caption("🗄️ **Storage:** Pinecone vector database")

if __name__ == "__main__":
    main()