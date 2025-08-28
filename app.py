import streamlit as st
import os
from dotenv import load_dotenv
from pdf_processor import PDFProcessor
from rag_system import RAGSystem

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

def main():
    st.set_page_config(
        page_title="RAG PDF Chatbot",
        page_icon="📚",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    st.title("📚 RAG PDF Chatbot")
    st.markdown("Upload PDFs and chat with their content using AI!")
    
    # Sidebar for PDF upload and settings
    with st.sidebar:
        st.header("📄 Document Management")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            key="pdf_uploader"
        )
        
        # Process uploaded files
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.processed_files:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
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
                            
                            st.success(f"✅ Processed {uploaded_file.name}")
                            
                        except Exception as e:
                            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
        
        # Display processed files
        if st.session_state.processed_files:
            st.subheader("📋 Processed Files")
            for filename in st.session_state.processed_files:
                st.text(f"• {filename}")
        
        # Clear memory button
        if st.button("🔄 Clear Chat Memory"):
            st.session_state.rag_system.clear_memory()
            st.session_state.messages = []
            st.success("Memory cleared!")
            st.experimental_rerun()
    
    # Main chat interface
    st.header("💬 Chat with your Documents")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            if not st.session_state.processed_files:
                response = "Please upload and process some PDF documents first!"
                st.markdown(response)
            else:
                with st.spinner("Thinking..."):
                    response = st.session_state.rag_system.query(prompt)
                    st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Footer
    st.markdown("---")
    st.markdown("**Note:** This chatbot uses OpenAI for vision processing, Groq for chat responses, and Pinecone for vector storage.")

if __name__ == "__main__":
    main()