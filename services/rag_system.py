# ===== services/enhanced_rag_system.py =====
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

# Groq LLM imports
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

logger = logging.getLogger(__name__)

class LocalRAGSystem:
    """RAG system optimized for vector-friendly content with display formatting preservation"""
    
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
        
        # Initialize Groq LLM
        self.llm = self._setup_groq_llm()
        
        # Text splitter for vector-optimized content
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Simple conversation history
        self.conversation_history = []
        self.max_history = 6  # Keep last 6 exchanges
    
    def _setup_groq_llm(self) -> Optional[Any]:
        """Setup Groq LLM"""
        if not GROQ_AVAILABLE:
            logger.warning("langchain-groq not available. Install with: pip install langchain-groq")
            return None
        
        if not Config.GROQ_API_KEY:
            logger.warning("GROQ_API_KEY not found. Please set your Groq API key.")
            return None
        
        try:
            llm = ChatGroq(
                api_key=Config.GROQ_API_KEY,
                model=Config.GROQ_MODEL,
                temperature=0.1
            )
            logger.info(f"Groq LLM loaded: {Config.GROQ_MODEL}")
            return llm
            
        except Exception as e:
            logger.error(f"Failed to load Groq LLM: {e}")
            logger.info("Check your Groq API key and internet connection")
            return None
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add processed documents to vector store with optimized content"""
        all_chunks = []
        all_metadatas = []
        
        for doc in documents:
            # Use vector-optimized content for embedding
            vector_content = doc.get("total_content", "")
            display_content = doc.get("display_content", vector_content)
            filename = doc.get("filename", "unknown")
            optimized_pages = doc.get("optimized_pages", [])
            
            if not vector_content.strip():
                continue
            
            # Process optimized pages individually for better chunking
            if optimized_pages:
                for page in optimized_pages:
                    page_vector_content = page.get("vector_content", "")
                    page_display_content = page.get("display_content", "")
                    page_num = page.get("page_number", 1)
                    content_metadata = page.get("content_metadata", [])
                    
                    if not page_vector_content.strip():
                        continue
                    
                    # Split page content into chunks
                    chunks = self.text_splitter.split_text(page_vector_content)
                    
                    for i, chunk in enumerate(chunks):
                        if chunk.strip():
                            all_chunks.append(chunk)
                            all_metadatas.append({
                                "filename": filename,
                                "page": page_num,
                                "chunk_index": i,
                                "total_chunks": len(chunks),
                                "extraction_method": doc.get("extraction_method", "unknown"),
                                "source_type": "vector_optimized",
                                "display_content": page_display_content,  # Store for display
                                "has_tables": any(meta.get("type") == "table" for meta in content_metadata),
                                "has_images": any(meta.get("type") == "image_text" for meta in content_metadata),
                                "content_types": [meta.get("type") for meta in content_metadata]
                            })
            else:
                # Fallback to basic chunking if no optimized pages
                chunks = self.text_splitter.split_text(vector_content)
                
                for i, chunk in enumerate(chunks):
                    if chunk.strip():
                        all_chunks.append(chunk)
                        all_metadatas.append({
                            "filename": filename,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "extraction_method": doc.get("extraction_method", "unknown"),
                            "source_type": "basic",
                            "display_content": display_content
                        })
        
        if all_chunks:
            # Add to FAISS vector store
            self.faiss_manager.add_texts(all_chunks, all_metadatas)
            logger.info(f"Added {len(all_chunks)} vector-optimized chunks to vector store")
        else:
            logger.warning("No valid content found to add to vector store")
    
    def query_documents(self, question: str) -> str:
        """Query documents with enhanced context formatting"""
        if not self.llm:
            return "Groq LLM not available. Please check your API key setup."
        
        try:
            # Search for relevant documents
            search_results = self.faiss_manager.similarity_search(question, k=4)
            
            if not search_results:
                return "I couldn't find relevant information in your documents for this question."
            
            # Prepare enhanced context
            context_chunks = []
            source_files = set()
            table_context = []
            
            for result in search_results:
                # Add the vector-optimized content (what was searched)
                context_chunks.append(result["text"])
                
                metadata = result["metadata"]
                if "filename" in metadata:
                    source_files.add(metadata["filename"])
                
                # If this chunk has tables, note it for special handling
                if metadata.get("has_tables", False):
                    table_context.append(f"This content includes table data from {metadata.get('filename', 'document')}, page {metadata.get('page', '?')}")
            
            document_context = "\n\n".join(context_chunks)
            conversation_context = self._format_context()
            
            # Enhanced prompt that acknowledges the content optimization
            prompt = f"""You are a helpful AI assistant analyzing documents. The content has been optimized for search but may contain natural language descriptions of tables and structured data.

CONVERSATION HISTORY:
{conversation_context}

DOCUMENT CONTEXT:
{document_context}

{f"SPECIAL NOTE: {'; '.join(table_context)}" if table_context else ""}

QUESTION: {question}

Instructions:
- Answer based on the document context provided
- When referencing data from tables, present it clearly even though the context shows natural language descriptions
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
                response += f"\n\nSources: {', '.join(source_files)}"
            
            # Update conversation history
            self._update_conversation(question, response)
            
            return response
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
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
        
        return "\n".join(context_parts)
    
    def chat_without_documents(self, message: str) -> str:
        """Handle normal conversation without documents"""
        if not self.llm:
            return "Groq LLM not available. Please check your API key setup."
        
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
            
            response_obj = self.llm.invoke(prompt)
            response = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
            
            self._update_conversation(message, response)
            
            return response
            
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _update_conversation(self, user_msg: str, ai_response: str):
        """Update conversation history"""
        self.conversation_history.append({
            "user": user_msg,
            "assistant": ai_response
        })
        
        
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def clear_memory(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        faiss_stats = self.faiss_manager.get_stats()
        
        return {
            "documents": faiss_stats.get("documents", 0),
            "conversation_length": len(self.conversation_history),
            "ocr_method": Config.OCR_METHOD,
            "llm_available": self.llm is not None,
            "vector_store": "FAISS (vector-optimized)"
        }