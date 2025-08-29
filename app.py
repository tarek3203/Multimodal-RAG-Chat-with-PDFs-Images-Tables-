# ===== app.py (Updated) =====
import streamlit as st
import logging
from pathlib import Path

from services.pdf_processor import PDFProcessor
from services.rag_system import LocalRAGSystem
from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = LocalRAGSystem()
    
    if 'pdf_processor' not in st.session_state:
        st.session_state.pdf_processor = PDFProcessor()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []

def process_uploaded_files(uploaded_files) -> bool:
    """Process uploaded PDF files"""
    success_count = 0
    
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.processed_files:
            try:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    # Read file bytes
                    pdf_bytes = uploaded_file.read()
                    
                    # Process with OCR-enabled PDF processor
                    processed_doc = st.session_state.pdf_processor.process_pdf(
                        pdf_bytes, uploaded_file.name
                    )
                    
                    # Add to RAG system
                    st.session_state.rag_system.add_documents([processed_doc])
                    
                    # Track processed files
                    st.session_state.processed_files.append(uploaded_file.name)
                    success_count += 1
                    
                    # Show processing details
                    method = processed_doc.get("extraction_method", "unknown")
                    content_length = processed_doc["metadata"]["content_length"]
                    st.success(f"✅ {uploaded_file.name} ({method}, {content_length} chars)")
                    
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                logger.error(f"Error processing {uploaded_file.name}: {e}")
    
    return success_count > 0

def get_ai_response(user_message: str, has_documents: bool) -> str:
    """Get AI response based on context"""
    if has_documents:
        return st.session_state.rag_system.query_documents(user_message)
    else:
        return st.session_state.rag_system.chat_without_documents(user_message)

def main():
    st.set_page_config(
        page_title="Local OCR + RAG Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    # Check if models directory exists
    if not Config.MODELS_FOLDER.exists():
        Config.ensure_directories()
    
    initialize_session_state()
    
    # Header
    st.title("🤖 Local OCR + RAG Chatbot")
    st.markdown("*Local AI with document analysis - no external APIs needed for core functionality*")
    
    # System status
    stats = st.session_state.rag_system.get_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📚 Documents", stats["documents"])
    with col2:
        st.metric("💬 Conversations", stats["conversation_length"])
    with col3:
        st.metric("🔍 OCR Method", stats["ocr_method"])
    with col4:
        st.metric("🧠 LLM", "✅ Local" if stats["llm_available"] else "❌ Missing")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Document Library")
        
        if st.session_state.processed_files:
            st.subheader("📋 Processed Documents")
            for i, filename in enumerate(st.session_state.processed_files, 1):
                st.text(f"{i}. {filename}")
        else:
            st.info("No documents uploaded yet")
        
        # Configuration
        st.subheader("⚙️ Configuration")
        current_ocr = st.selectbox(
            "OCR Method",
            ["trocr", "easyocr", "got_ocr"],
            index=["trocr", "easyocr", "got_ocr"].index(Config.OCR_METHOD)
        )
        
        if st.button("🔄 Restart OCR Processor"):
            # Reinitialize with new OCR method
            Config.OCR_METHOD = current_ocr
            st.session_state.pdf_processor = PDFProcessor()
            st.success(f"Switched to {current_ocr}")
        
        # Clear functions
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.rag_system.clear_memory()
            st.session_state.messages = []
            st.success("✅ Chat cleared!")
            st.rerun()
        
        if st.button("🗑️ Clear Documents", use_container_width=True):
            st.session_state.rag_system.faiss_manager.delete_index()
            st.session_state.processed_files = []
            st.success("✅ Documents cleared!")
            st.rerun()
    
    # File upload
    st.subheader("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Upload PDFs for analysis. OCR method: {Config.OCR_METHOD}"
    )
    
    if uploaded_files:
        if st.button("📥 Process Files", use_container_width=True):
            if process_uploaded_files(uploaded_files):
                st.rerun()
    
    # Chat interface
    st.subheader("💬 Chat")
    
    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        has_documents = len(st.session_state.processed_files) > 0
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_ai_response(prompt, has_documents)
                st.markdown(response)
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.rerun()
    
    # Footer
    st.markdown("""
    **System Info:**
    - OCR: Local models (TrOCR/EasyOCR/GOT-OCR)
    - LLM: Groq (Llama3-70B-8192)
    - Vector Store: FAISS (local storage)
    - Embeddings: HuggingFace sentence-transformers
    """)

    # Update the warning message:
    if not stats["llm_available"]:
        st.warning("""
        **Groq LLM not available!** 
        
        Get your free API key from Groq:
        1. Visit: https://console.groq.com/
        2. Sign up and get your API key
        3. Set environment variable: export GROQ_API_KEY="your-key-here"
        
        Or create a .env file:
        ```
        GROQ_API_KEY=your-groq-api-key-here
        ```
        """)

if __name__ == "__main__":
    main()