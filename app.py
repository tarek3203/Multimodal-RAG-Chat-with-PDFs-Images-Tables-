# ===== app.py =====
"""
Simplified multimodal RAG chatbot with PDF upload and chat interface
"""
import streamlit as st
import logging
from pathlib import Path
import time
import asyncio

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
    
    if 'docs_cleared' not in st.session_state:
        st.session_state.docs_cleared = False

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

def get_ai_response_stream(user_message: str, has_documents: bool):
    """Get streaming AI response based on context"""
    if has_documents:
        return st.session_state.rag_system.query_documents_stream(user_message)
    else:
        return st.session_state.rag_system.chat_without_documents_stream(user_message)

def main():
    # Ensure directories exist
    if not Config.MODELS_FOLDER.exists():
        Config.ensure_directories()
    
    initialize_session_state()
    
    # Sidebar for PDF uploads and controls
    with st.sidebar:
        st.title("📄 Document Upload")
        
        # System status - only show document count
        stats = st.session_state.rag_system.get_stats()
        st.metric("Documents", stats["documents_in_vector_store"])
        
        st.divider()
        
        # File upload section
        st.subheader("📁 Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files to analyze",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload PDF documents for analysis. Files will be processed automatically."
        )
        
        # Auto-process files when uploaded (but not if docs were just cleared)
        if uploaded_files and not st.session_state.docs_cleared:
            # Check if any new files need processing
            new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
            if new_files:
                if process_uploaded_files(new_files):
                    st.rerun()
        
        # Reset the cleared flag if user uploads new files after clearing
        if uploaded_files and st.session_state.docs_cleared:
            st.session_state.docs_cleared = False
        
        # Show processed files
        if st.session_state.processed_files:
            st.subheader("📚 Processed Documents")
            for i, filename in enumerate(st.session_state.processed_files, 1):
                st.text(f"{i}. {filename}")
        
        st.divider()
        
        # Clear options - only keep chat and docs clearing
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
                st.session_state.docs_cleared = True  # Set flag to prevent auto-processing
                st.success("Documents cleared!")
                st.rerun()
    
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
            
            # Get AI response with streaming and completion handling
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                
                try:
                    # Stream the response with fast typing speed (100 WPM)
                    for chunk in get_ai_response_stream(prompt, has_documents):
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                        
                        # Add natural typing delay for 100 WPM (very fast pace)
                        # 100 WPM = 100 words / 60 seconds = 1.67 words per second
                        # So 1 word takes 0.6 seconds (60/100)
                        words_in_chunk = len(chunk.split())
                        delay = min(words_in_chunk * 0.06, 0.25)  # Max 0.25s delay, very fast pace
                        if delay > 0:
                            time.sleep(delay)
                    
                    # Final response without cursor
                    response_placeholder.markdown(full_response)
                    
                except Exception as stream_error:
                    st.error(f"Streaming error: {stream_error}")
                    # Fallback to non-streaming
                    try:
                        fallback_response = get_ai_response(prompt, has_documents)
                        response_placeholder.markdown(fallback_response)
                        full_response = fallback_response
                    except Exception as fallback_error:
                        error_msg = "Sorry, I encountered an error. Please try again."
                        response_placeholder.markdown(error_msg)
                        full_response = error_msg
            
            # Add assistant response
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
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

if __name__ == "__main__":
    main()