# ===== services/rag_system.py =====
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Add project root to path to import vector_services
sys.path.append(str(Path(__file__).parent.parent))

from vector_services.embeddings import EmbeddingService
from vector_services.faiss_manager import FAISSManager
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import Config

# Local LLM imports
try:
    from langchain_ollama import ChatOllama
    LLAMA_OLLAMA_AVAILABLE = True
except ImportError:
    LLAMA_OLLAMA_AVAILABLE = False

logger = logging.getLogger(__name__)

class LocalRAGSystem:
    """Local RAG system using FAISS and local LLM"""
    
    def __init__(self, index_name: str = "pdf_documents"):
        Config.ensure_directories()
        
        # Initialize embedding service
        self.embedding_service = EmbeddingService(Config.EMBEDDING_MODEL)
        
        # Initialize FAISS manager
        self.faiss_manager = FAISSManager(
            persist_directory=str(Config.VECTOR_FOLDER),
            index_name=index_name,
            embedding_service=self.embedding_service
        )
        
        # Initialize local LLM
        self.llm = self._setup_local_llm()
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\\n\\n", "\\n", " ", ""]
        )
        
        # Simple conversation history
        self.conversation_history = []
        self.max_history = 6  # Keep last 6 exchanges
    
    def _setup_local_llm(self) -> Optional[Any]:
        """Setup local LLM using ChatOllama"""
        if not LLAMA_OLLAMA_AVAILABLE:
            logger.warning("langchain-ollama not available. Install with: pip install langchain-ollama")
            return None
        
        try:
            # ChatOllama uses model names, not file paths
            llm = ChatOllama(
                model="llama2:7b-chat",  # Ollama model name
                temperature=0.1
            )
            logger.info("✅ Local LLM loaded with ChatOllama (llama2:7b-chat)")
            return llm
            
        except Exception as e:
            logger.error(f"Failed to load ChatOllama: {e}")
            logger.info("Make sure Ollama is running: ollama serve")
            logger.info("And that you have the model: ollama pull llama2:7b-chat")
            return None
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add processed documents to the vector store"""
        all_chunks = []
        all_metadatas = []
        
        for doc in documents:
            content = doc.get("total_content", "")
            filename = doc.get("filename", "unknown")
            
            if not content.strip():
                continue
            
            # Split content into chunks
            chunks = self.text_splitter.split_text(content)
            
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    all_chunks.append(chunk)
                    all_metadatas.append({
                        "filename": filename,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "extraction_method": doc.get("extraction_method", "unknown"),
                        "source_type": "pdf_content"
                    })
        
        if all_chunks:
            # Add to FAISS vector store
            self.faiss_manager.add_texts(all_chunks, all_metadatas)
            logger.info(f"✅ Added {len(all_chunks)} chunks to vector store")
        else:
            logger.warning("No valid content found to add to vector store")
    
    def _format_context(self) -> str:
        """Format recent conversation history"""
        if not self.conversation_history:
            return ""
        
        context_parts = []
        for exchange in self.conversation_history[-3:]:  # Last 3 exchanges
            context_parts.extend([
                f"Human: {exchange['user']}",
                f"Assistant: {exchange['assistant']}"
            ])
        
        return "\\n".join(context_parts)
    
    def query_documents(self, question: str) -> str:
        """Query the document collection using RAG"""
        if not self.llm:
            return "❌ Local LLM not available. Please check your model setup."
        
        try:
            # Search for relevant documents
            search_results = self.faiss_manager.similarity_search(question, k=3)
            
            if not search_results:
                return "❓ I couldn't find relevant information in your documents for this question."
            
            # Prepare context
            context_chunks = []
            source_files = set()
            
            for result in search_results:
                context_chunks.append(result["text"])
                if "filename" in result["metadata"]:
                    source_files.add(result["metadata"]["filename"])
            
            document_context = "\\n\\n".join(context_chunks)
            conversation_context = self._format_context()
            
            # Generate response
            prompt = f"""You are a helpful AI assistant analyzing documents. Answer the question based on the provided context.

            CONVERSATION HISTORY:
            {conversation_context}

            DOCUMENT CONTEXT:
            {document_context}

            QUESTION: {question}

            Instructions:
            - Answer based primarily on the document context
            - Be specific and cite relevant information
            - If the answer isn't in the documents, say so clearly
            - Maintain conversation continuity
            - Be concise but thorough

            Answer:"""
            
            # Use invoke() and get content from response
            response_obj = self.llm.invoke(prompt)
            response = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
            
            # Add source information
            if source_files:
                response += f"\\n\\n📄 *Sources: {', '.join(source_files)}*"
            
            # Update conversation history
            self._update_conversation(question, response)
            
            return response
            
        except Exception as e:
            error_msg = f"❌ Error processing question: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def chat_without_documents(self, message: str) -> str:
        """Handle normal conversation without documents"""
        if not self.llm:
            return "❌ Local LLM not available. Please check your model setup."
        
        try:
            conversation_context = self._format_context()
            
            prompt = f"""You are a helpful AI assistant having a natural conversation.

    CONVERSATION HISTORY:
    {conversation_context}

    MESSAGE: {message}

    Instructions:
    - Respond naturally and conversationally
    - Reference previous conversation when relevant
    - Be helpful and informative
    - If asked about documents, explain that no documents are currently loaded

    Response:"""
            
            # Use invoke() and get content from response
            response_obj = self.llm.invoke(prompt)
            response = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
            
            # Update conversation history
            self._update_conversation(message, response)
            
            return response
            
        except Exception as e:
            error_msg = f"❌ Error processing message: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _update_conversation(self, user_msg: str, ai_response: str):
        """Update conversation history"""
        self.conversation_history.append({
            "user": user_msg,
            "assistant": ai_response
        })
        
        # Keep only recent history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def clear_memory(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("🧹 Conversation history cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        faiss_stats = self.faiss_manager.get_stats()
        
        return {
            "documents": faiss_stats.get("documents", 0),
            "conversation_length": len(self.conversation_history),
            "ocr_method": Config.OCR_METHOD,
            "llm_available": self.llm is not None,
            "vector_store": "FAISS (local)"
        }