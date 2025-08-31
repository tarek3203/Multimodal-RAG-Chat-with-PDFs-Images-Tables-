# ===== app.py =====
"""
Simplified multimodal RAG chatbot with PDF upload and chat interface
"""
import streamlit as st
import logging
from pathlib import Path
import time

from services.pdf_processor import MultimodalPDFProcessor
from services.rag_system import MultimodalRAGSystem
from config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Multimodal PDF RAG Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = MultimodalRAGSystem()
    
    if 'pdf_processor' not in st.session_state:
        st.session_state.pdf_processor = MultimodalPDFProcessor()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = None

def process_uploaded_files(uploaded_files) -> bool:
    """Process uploaded PDF files with progress tracking"""
    success_count = 0
    
    # Create progress tracking
    progress_container = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        if uploaded_file.name not in st.session_state.processed_files:
            try:
                # Update progress
                progress_container.info(f"Processing {uploaded_file.name}... ({i+1}/{len(uploaded_files)})")
                
                # Read file bytes
                pdf_bytes = uploaded_file.read()
                
                # Process with multimodal PDF processor
                with st.spinner(f"Extracting content from {uploaded_file.name}..."):
                    processed_doc = st.session_state.pdf_processor.process_pdf(
                        pdf_bytes, uploaded_file.name
                    )
                
                # Add to RAG system
                with st.spinner(f"Adding {uploaded_file.name} to knowledge base..."):
                    st.session_state.rag_system.add_documents([processed_doc])
                
                # Track processed files
                st.session_state.processed_files.append(uploaded_file.name)
                success_count += 1
                
                # Show processing details
                metadata = processed_doc.get("metadata", {})
                extraction_method = metadata.get("extraction_method", "unknown")
                content_length = metadata.get("content_length", 0)
                
                # Content breakdown
                text_count = metadata.get("text_count", 0)
                table_count = metadata.get("table_count", 0)
                image_count = metadata.get("image_count", 0)
                
                details = []
                if text_count > 0:
                    details.append(f"{text_count} text sections")
                if table_count > 0:
                    details.append(f"{table_count} tables")
                if image_count > 0:
                    details.append(f"{image_count} images")
                
                detail_text = f" ({', '.join(details)})" if details else ""
                
                progress_container.success(
                    f"✅ {uploaded_file.name} processed successfully "
                    f"({extraction_method}, {content_length:,} chars{detail_text})"
                )
                
                # Brief pause to show success message
                time.sleep(0.5)
                
            except Exception as e:
                progress_container.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                logger.error(f"Error processing {uploaded_file.name}: {e}")
                time.sleep(1)
    
    # Clear progress container
    progress_container.empty()
    
    return success_count > 0

def get_ai_response(user_message: str, has_documents: bool) -> str:
    """Get AI response based on context"""
    if has_documents:
        return st.session_state.rag_system.query_documents(user_message)
    else:
        return st.session_state.rag_system.chat_without_documents(user_message)

def main():
    # Ensure directories exist
    if not Config.MODELS_FOLDER.exists():
        Config.ensure_directories()
    
    initialize_session_state()
    
    # Sidebar for PDF uploads and controls
    with st.sidebar:
        st.title("📄 Document Upload")
        
        # System status
        stats = st.session_state.rag_system.get_stats()
        
        # Status metrics
        st.subheader("📊 System Status")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", stats["documents_in_vector_store"])
            st.metric("LLM", "✅ Ready" if stats["llm_available"] else "❌ Missing")
        
        with col2:
            st.metric("Conversations", stats["conversation_length"])
            st.metric("Embedding", "✅ Ready")
        
        # Content breakdown if available
        if stats.get("content_types"):
            st.subheader("📋 Content Types")
            content_types = stats["content_types"]
            for content_type, count in content_types.items():
                if count > 0:
                    st.text(f"{content_type.title()}: {count}")
        
        st.divider()
        
        # File upload section
        st.subheader("📁 Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files to analyze",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload PDF documents for analysis. The system will extract text, tables, and images."
        )
        
        if uploaded_files:
            if st.button("🚀 Process Files", use_container_width=True, type="primary"):
                if process_uploaded_files(uploaded_files):
                    st.rerun()
        
        # Show processed files
        if st.session_state.processed_files:
            st.subheader("📚 Processed Documents")
            for i, filename in enumerate(st.session_state.processed_files, 1):
                st.text(f"{i}. {filename}")
        
        st.divider()
        
        # Clear options
        st.subheader("🧹 Clear Data")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗨️ Clear Chat", use_container_width=True):
                st.session_state.rag_system.clear_memory()
                st.session_state.messages = []
                st.success("Chat cleared!")
                st.rerun()
        
        with col2:
            if st.button("📄 Clear Docs", use_container_width=True):
                st.session_state.rag_system.clear_documents()
                st.session_state.processed_files = []
                st.success("Documents cleared!")
                st.rerun()
        
        # System information
        st.divider()
        st.subheader("ℹ️ System Info")
        st.text(f"Vector Store: {stats['vector_store_type']}")
        st.text(f"Embedding Model: {stats['embedding_model'].split('/')[-1]}")
        
        # API Status
        api_status = []
        if Config.GROQ_API_KEY:
            api_status.append("✅ Groq (Chat)")
        if Config.GOOGLE_API_KEY:
            api_status.append("✅ Google (Vision)")
        elif Config.OPENAI_API_KEY:
            api_status.append("✅ OpenAI (Vision)")
        else:
            api_status.append("❌ Vision API Missing")
        
        for status in api_status:
            st.text(status)
    
    # Main chat interface
    st.title("🤖 Multimodal PDF RAG Chatbot")
    st.markdown("*Chat with your documents using advanced AI - supports text, tables, and images*")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about your documents..."):
            has_documents = len(st.session_state.processed_files) > 0
            
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
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
    
    # Footer information
    if not st.session_state.processed_files:
        st.info(
            "👆 **Get started by uploading PDF documents in the sidebar!**\n\n"
            "This system can analyze:\n"
            "- 📝 Text content from PDFs\n"
            "- 📊 Tables and structured data\n"
            "- 🖼️ Images with OCR and visual analysis\n"
            "- 💬 Multiple document conversations"
        )
    
    # System architecture info (expandable)
    with st.expander("🔧 System Architecture", expanded=False):
        st.markdown("""
        **Enhanced Multimodal RAG Pipeline:**
        
        1. **PDF Processing**: Unstructured library extracts text, tables, and images
        2. **Content Analysis**: 
           - Text & Tables: Groq Llama-3.1 summarization
           - Images: Google Gemini or OpenAI GPT-4V vision analysis
        3. **Vector Storage**: FAISS local vector database with HuggingFace embeddings
        4. **Retrieval**: Semantic search across all content types
        5. **Generation**: Groq Llama-3.1 with multimodal context synthesis
        
        **Key Features:**
        - Local vector storage (no data sent to external vector DBs)
        - Multimodal understanding (text, tables, images)
        - Conversation memory and context
        - Source attribution and content type tracking
        """)

if __name__ == "__main__":
    main()