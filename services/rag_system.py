# ===== services/rag_system.py =====
"""
Enhanced multimodal RAG system with FAISS vector store and document management
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
# from xml.dom.minidom import Document
import uuid
from langchain.schema.document import Document
from langchain_community.vectorstores import FAISS
# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from vector_services.embeddings import MultimodalEmbeddingService
from vector_services.faiss_manager import FAISSManager
from prompts import PromptTemplates
from config import Config

logger = logging.getLogger(__name__)

class MultimodalRAGSystem:
    """Enhanced RAG system for multimodal document processing and querying"""
    
    def __init__(self, index_name: str = "multimodal_documents"):
        Config.ensure_directories()
        
        self.config = Config
        
        # Initialize embedding service
        self.embedding_service = MultimodalEmbeddingService(Config.EMBEDDING_MODEL)
        
        # Initialize FAISS manager
        self.faiss_manager = FAISSManager(
            persist_directory=str(Config.VECTOR_FOLDER),
            index_name=index_name,
            embedding_service=self.embedding_service
        )
        
        # Initialize LLM for generation
        self.llm = self._setup_groq_llm()
        
        # Document store for original content
        self.doc_store = {}
        
        # Conversation history
        self.conversation_history = []
        self.max_history = 6  # Keep last 6 exchanges
        
        logger.info("Multimodal RAG system initialized")
    
    def _setup_groq_llm(self) -> Optional[Any]:
        """Setup Groq LLM for generation"""
        try:
            from langchain_groq import ChatGroq

            
            if not Config.GROQ_API_KEY:
                logger.warning("GROQ_API_KEY not found. Please set your Groq API key.")
                return None
            
            llm = ChatGroq(
                api_key=Config.GROQ_API_KEY,
                model=Config.GROQ_MODEL,
                temperature=0.1
            )
            logger.info(f"Groq LLM loaded: {Config.GROQ_MODEL}")
            return llm
            
        except ImportError:
            logger.warning("langchain-groq not available. Install with: pip install langchain-groq")
            return None
        except Exception as e:
            logger.error(f"Failed to load Groq LLM: {e}")
            return None
    
    def add_documents(self, processed_documents: List[Dict[str, Any]]):
        """
        Add documents following the exact colab notebook pattern
        """
        if not processed_documents:
            logger.warning("No documents provided")
            return
        
        try:
            for processed_doc in processed_documents:
                filename = processed_doc.get("filename", "unknown")
                
                # Get the summaries and original content
                text_summaries = processed_doc.get("text_summaries", [])
                table_summaries = processed_doc.get("table_summaries", [])  
                image_summaries = processed_doc.get("image_summaries", [])
                
                texts = processed_doc.get("texts", [])
                tables = processed_doc.get("tables", [])
                images = processed_doc.get("images", [])
                
                # Add texts (following colab pattern exactly)
                if text_summaries and texts:
                    doc_ids = [str(uuid.uuid4()) for _ in texts]
                    summary_texts = [
                        Document(page_content=summary, metadata={self.faiss_manager.id_key: doc_ids[i], "filename": filename, "type": "text"}) 
                        for i, summary in enumerate(text_summaries)
                    ]
                    
                    
                    self.retriever.vectorstore = FAISS.from_documents(summary_texts, self.faiss_manager.embeddings)
                    self.faiss_manager.vectorstore = self.retriever.vectorstore
                    
                    self.retriever.docstore.mset(list(zip(doc_ids, texts)))
                    logger.info(f"Added {len(texts)} text elements from {filename}")
                
                # Add tables (following colab pattern exactly)
                if table_summaries and tables:
                    table_ids = [str(uuid.uuid4()) for _ in tables]
                    summary_tables = [
                        Document(page_content=summary, metadata={self.faiss_manager.id_key: table_ids[i], "filename": filename, "type": "table"}) 
                        for i, summary in enumerate(table_summaries)
                    ]
                    
                    self.retriever.vectorstore = FAISS.from_documents(summary_tables, self.faiss_manager.embeddings)
                    self.faiss_manager.vectorstore = self.retriever.vectorstore
                    
                    self.retriever.docstore.mset(list(zip(table_ids, tables)))
                    logger.info(f"Added {len(tables)} table elements from {filename}")
                
                # Add image summaries (following colab pattern exactly)
                if image_summaries:
                    img_ids = [str(uuid.uuid4()) for _ in image_summaries]
                    summary_img = [
                        Document(page_content=summary, metadata={self.faiss_manager.id_key: img_ids[i], "filename": filename, "type": "image"}) for i, summary in enumerate(image_summaries)
                    ]

                    self.retriever.vectorstore = FAISS.from_documents(summary_img, self.faiss_manager.embeddings)
                    self.faiss_manager.vectorstore = self.retriever.vectorstore
                    
                    # For images, store the descriptions as the "original" content
                    self.retriever.docstore.mset(list(zip(img_ids, images)))
                    logger.info(f"Added {len(image_summaries)} image elements from {filename}")
            
            # Save the index after adding all documents
            self.faiss_manager._save_index()
            logger.info("Successfully added all documents to RAG system")
            
        except Exception as e:
            logger.error(f"Error adding documents to RAG system: {e}")
            raise
    
    def query_documents(self, question: str) -> str:
        """Query the RAG system - simple and direct"""
        if not self.llm:
            return "LLM not available. Please check your GROQ_API_KEY."
        
        # Retrieve relevant documents using the retriever (exact colab pattern)
        docs = self.retriever.invoke(question)
        
        if not docs:
            return "I couldn't find relevant information for your question."
        
        # Format context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Format conversation history
        history = ""
        if self.conversation_history:
            recent_history = self.conversation_history[-3:]  # Last 3 exchanges
            history_parts = []
            for exchange in recent_history:
                history_parts.extend([
                    f"Human: {exchange['user']}",
                    f"Assistant: {exchange['assistant']}"
                ])
            history = "\n".join(history_parts)
        
            # Simple prompt
            prompt = f"""Answer the question based on the provided context and conversation history.

            Conversation History:
            {history}

            Context:
            {context}

            Question: {question}

            Answer:"""
        
        # Generate response
        try:
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Update conversation history
            self.conversation_history.append({
                "user": question,
                "assistant": response_text
            })
            
            # Keep only last 6 exchanges
            if len(self.conversation_history) > 6:
                self.conversation_history = self.conversation_history[-6:]
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {e}"
    
    def chat_without_documents(self, message: str) -> str:
        """Handle normal conversation without documents"""
        if not self.llm:
            return "Language model not available. Please check your API configuration."
        
        try:
            conversation_context = self._format_conversation_context()
            prompt = PromptTemplates.get_general_chat_prompt()
            
            response_obj = self.llm.invoke(prompt.format(
                conversation_context=conversation_context,
                message=message
            ))
            
            response = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
            
            self._update_conversation(message, response)
            return response
            
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _format_conversation_context(self) -> str:
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
    
    def clear_documents(self):
        """Clear all documents from vector store and doc store"""
        try:
            self.faiss_manager.delete_index()
            self.doc_store = {}
            logger.info("All documents cleared from RAG system")
        except Exception as e:
            logger.error(f"Error clearing documents: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        faiss_stats = self.faiss_manager.get_stats()
        
        return {
            "documents_in_vector_store": faiss_stats.get("documents", 0),
            "documents_in_doc_store": len(self.doc_store),
            "conversation_length": len(self.conversation_history),
            "llm_available": self.llm is not None,
            "vector_store_type": "FAISS",
            "embedding_model": self.embedding_service.model_name,
            "content_types": self._get_content_type_stats()
        }
    
    def _get_content_type_stats(self) -> Dict[str, int]:
        """Get statistics about content types in doc store"""
        content_types = {"text": 0, "table": 0, "image": 0}
        
        for doc_data in self.doc_store.values():
            content_type = doc_data.get("type", "unknown")
            if content_type in content_types:
                content_types[content_type] += 1
        
        return content_types


# Legacy compatibility 
LocalRAGSystem = MultimodalRAGSystem