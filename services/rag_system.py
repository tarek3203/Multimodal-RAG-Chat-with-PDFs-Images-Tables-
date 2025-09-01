# ===== services/rag_system.py =====
"""
Enhanced multimodal RAG system with FAISS vector store and document management
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import uuid
import base64
import time
from langchain.schema.document import Document
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from vector_services.embeddings import get_embeddings
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
        self.embedding_service = get_embeddings(Config.EMBEDDING_MODEL)
        
        # Initialize FAISS manager
        self.faiss_manager = FAISSManager(
            embedding_model=Config.EMBEDDING_MODEL
        )
        
        # Set up retriever from faiss_manager
        self.retriever = self.faiss_manager.retriever
        
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
                temperature=0.1,
                streaming=True,
                verbose=False,
                max_tokens=500,  # Ensure adequate token limit
                max_retries=3,    # Retry on failures
                request_timeout=120  # 2 minute timeout
            )
            logger.info(f"Groq LLM loaded: {Config.GROQ_MODEL}")
            return llm
            
        except ImportError:
            logger.warning("langchain-groq not available. Install with: pip install langchain-groq")
            return None
        except Exception as e:
            logger.error(f"Failed to load Groq LLM: {e}")
            return None
    
    def _parse_retrieved_docs(self, docs: List[Any]) -> Dict[str, List[str]]:
        """Parse retrieved documents to separate base64 images from text content"""
        b64_images = []
        text_content = []
        
        for doc in docs:
            doc_str = str(doc)
            try:
                # Try to decode as base64 - if successful, it's an image
                base64.b64decode(doc_str, validate=True)
                # Check if it looks like a base64 image (common patterns)
                if doc_str.startswith('/9j/') or doc_str.startswith('iVBORw0KGgo') or len(doc_str) > 1000:
                    b64_images.append(doc_str)
                else:
                    text_content.append(doc_str)
            except Exception:
                # If base64 decode fails, it's text content
                text_content.append(doc_str)
        
        return {"images": b64_images, "texts": text_content}
    
    def _build_multimodal_prompt(self, question: str, parsed_docs: Dict[str, List[str]], 
                                conversation_history: str = "") -> str:
        """Build multimodal prompt using enhanced templates from prompts.py"""
        
        # Combine all text content
        context_text = "\n\n".join(parsed_docs["texts"])
        
        # Use enhanced prompts from prompts.py
        if conversation_history:
            prompt_template = PromptTemplates.get_enhanced_rag_query_prompt()
            prompt_text = prompt_template.format(
                conversation_history=conversation_history,
                context_text=context_text,
                question=question
            )
        else:
            prompt_template = PromptTemplates.get_enhanced_rag_query_simple_prompt()
            prompt_text = prompt_template.format(
                context_text=context_text,
                question=question
            )
        
        return prompt_text
    
    def add_documents(self, processed_documents: List[Dict[str, Any]]):
        """
        Add documents properly - accumulate all content types like the notebook
        """
        if not processed_documents:
            logger.warning("No documents provided")
            return
        
        try:
            # Collect ALL documents before adding them
            all_summary_docs = []
            all_doc_ids = []
            all_original_content = []
            
            for processed_doc in processed_documents:
                filename = processed_doc.get("filename", "unknown")
                
                # Get the summaries and original content
                text_summaries = processed_doc.get("text_summaries", [])
                table_summaries = processed_doc.get("table_summaries", [])  
                image_summaries = processed_doc.get("image_summaries", [])
                
                texts = processed_doc.get("texts", [])
                tables = processed_doc.get("tables", [])
                images = processed_doc.get("images", [])
                
                # 📝 Add text elements
                if text_summaries and texts:
                    doc_ids = [str(uuid.uuid4()) for _ in texts]
                    summary_docs = [
                        Document(page_content=summary, metadata={self.faiss_manager.id_key: doc_ids[i], "filename": filename, "type": "text"}) 
                        for i, summary in enumerate(text_summaries)
                    ]
                    
                    all_summary_docs.extend(summary_docs)
                    all_doc_ids.extend(doc_ids)
                    all_original_content.extend(texts)
                    logger.info(f"Prepared {len(texts)} text elements from {filename}")
                
                # 📊 Add table elements  
                if table_summaries and tables:
                    table_ids = [str(uuid.uuid4()) for _ in tables]
                    summary_docs = [
                        Document(page_content=summary, metadata={self.faiss_manager.id_key: table_ids[i], "filename": filename, "type": "table"}) 
                        for i, summary in enumerate(table_summaries)
                    ]
                    
                    all_summary_docs.extend(summary_docs)
                    all_doc_ids.extend(table_ids)
                    all_original_content.extend(tables)
                    logger.info(f"Prepared {len(tables)} table elements from {filename}")
                
                # 🖼️ Add image elements
                if image_summaries:
                    img_ids = [str(uuid.uuid4()) for _ in image_summaries]
                    summary_docs = [
                        Document(page_content=summary, metadata={self.faiss_manager.id_key: img_ids[i], "filename": filename, "type": "image"}) 
                        for i, summary in enumerate(image_summaries)
                    ]
                    
                    all_summary_docs.extend(summary_docs)
                    all_doc_ids.extend(img_ids)
                    all_original_content.extend(image_summaries)  # Store summaries, not base64
                    logger.info(f"Prepared {len(image_summaries)} image elements from {filename}")
            
            # 🔧 Now add ALL documents at once (like the notebook)
            if all_summary_docs:
                # Check if vectorstore already has content
                current_doc_count = 0
                try:
                    if hasattr(self.retriever.vectorstore, 'index') and self.retriever.vectorstore.index:
                        current_doc_count = self.retriever.vectorstore.index.ntotal
                except:
                    current_doc_count = 0
                
                if current_doc_count > 1:  # Has existing content (dummy doc = 1)
                    # Add to existing vectorstore
                    self.retriever.vectorstore.add_documents(all_summary_docs)
                    logger.info(f"Added {len(all_summary_docs)} documents to existing vectorstore")
                else:
                    # Create new vectorstore with all documents
                    self.retriever.vectorstore = FAISS.from_documents(all_summary_docs, self.faiss_manager.embeddings)
                    logger.info(f"Created new vectorstore with {len(all_summary_docs)} documents")
                
                # Update faiss_manager reference
                self.faiss_manager.vectorstore = self.retriever.vectorstore
                
                # Add all original content to docstore
                self.retriever.docstore.mset(list(zip(all_doc_ids, all_original_content)))
                
                logger.info(f"Successfully added {len(all_summary_docs)} total elements to RAG system")
                logger.info(f"Vector store now contains {self.retriever.vectorstore.index.ntotal} documents")
            
        except Exception as e:
            logger.error(f"Error adding documents to RAG system: {e}")
            raise
    
    def query_documents_stream(self, question: str):
        """Query the RAG system with streaming response"""
        if not self.llm:
            yield "LLM not available. Please check your GROQ_API_KEY."
            return
        
        try:
            # Retrieve relevant documents using the retriever
            docs = self.retriever.invoke(question)
            
            if not docs:
                yield "I couldn't find relevant information for your question."
                return
            
            # Parse documents to separate images from text
            parsed_docs = self._parse_retrieved_docs(docs)
            
            # Format conversation history
            history = ""
            if self.conversation_history:
                recent_history = self.conversation_history[-3:]
                history_parts = []
                for exchange in recent_history:
                    history_parts.extend([
                        f"Human: {exchange['user']}",
                        f"Assistant: {exchange['assistant']}"
                    ])
                history = "\n".join(history_parts)
            
            # Build prompt using parsed content
            prompt = self._build_multimodal_prompt(question, parsed_docs, history)
            
            # Generate streaming response with natural pacing and completion tracking
            response_text = ""
            buffer = ""
            word_count = 0
            last_chunk_time = time.time()
            chunks_received = 0
            
            try:
                for chunk in self.llm.stream(prompt):
                    if hasattr(chunk, 'content') and chunk.content:
                        response_text += chunk.content
                        buffer += chunk.content
                        chunks_received += 1
                        last_chunk_time = time.time()
                        
                        # Count words to control release timing
                        word_count += len(chunk.content.split())
                        
                        # Release buffer when we have enough content (every 2-4 words)
                        if word_count >= 2 or buffer.endswith(('.', '!', '?', '\n')):
                            yield buffer
                            buffer = ""
                            word_count = 0
                    
                    # Check for timeout or stalled streaming
                    elif time.time() - last_chunk_time > 30:  # 30 second timeout
                        logger.warning("Stream timeout detected, attempting completion")
                        break
                
                # Yield any remaining content
                if buffer:
                    yield buffer
                
                # Check if response seems incomplete and try to complete it
                if self._is_response_incomplete(response_text) and chunks_received > 5:
                    logger.info("Response appears incomplete, attempting to complete...")
                    completion = self._attempt_completion(response_text, question)
                    if completion:
                        yield f"\n\n{completion}"
                        response_text += f"\n\n{completion}"
                        
            except Exception as stream_error:
                logger.error(f"Streaming error: {stream_error}")
                # Fallback to non-streaming if streaming fails
                try:
                    fallback_response = self.llm.invoke(prompt)
                    if hasattr(fallback_response, 'content'):
                        response_text = fallback_response.content
                        yield response_text
                except Exception as fallback_error:
                    logger.error(f"Fallback error: {fallback_error}")
                    yield "Error: Unable to generate complete response. Please try again."
                    return
            
            # Update conversation history with complete response
            self.conversation_history.append({
                "user": question,
                "assistant": response_text
            })
            
            # Keep only last 6 exchanges
            if len(self.conversation_history) > 6:
                self.conversation_history = self.conversation_history[-6:]
                
        except Exception as e:
            logger.error(f"Error generating streaming response: {e}")
            yield f"Error generating response: {e}"

    def query_documents(self, question: str) -> str:
        """Query the RAG system with proper document parsing"""
        if not self.llm:
            return "LLM not available. Please check your GROQ_API_KEY."
        
        try:
            # Retrieve relevant documents using the retriever
            docs = self.retriever.invoke(question)
            
            if not docs:
                return "I couldn't find relevant information for your question."
            
            # 🔍 DEBUG: Print raw retrieved docs to understand the issue
            print("=" * 80)
            print("📋 RAW RETRIEVED DOCUMENTS DEBUG")
            print("=" * 80)
            print(f"📊 Number of documents retrieved: {len(docs)}")
            for i, doc in enumerate(docs):
                doc_str = str(doc)
                print(f"\n📄 Document {i+1}:")
                print(f"  Type: {type(doc)}")
                print(f"  Length: {len(doc_str)}")
                print(f"  First 100 chars: {doc_str[:100]}...")
                
                # Check if it's base64
                try:
                    base64.b64decode(doc_str[:100], validate=True)
                    print(f"  🖼️  Detected as: BASE64 IMAGE")
                except:
                    print(f"  � Detected as: TEXT CONTENT")
            print("=" * 80)
            
            # Parse documents to separate images from text
            parsed_docs = self._parse_retrieved_docs(docs)
            
            print(f"📊 PARSED RESULTS:")
            print(f"  📝 Text documents: {len(parsed_docs['texts'])}")
            print(f"  �️  Image documents: {len(parsed_docs['images'])}")
            
            # Show text content
            if parsed_docs['texts']:
                print(f"\n📝 TEXT CONTENT:")
                for i, text in enumerate(parsed_docs['texts']):
                    print(f"  Text {i+1}: {text[:200]}...")
            
            print("=" * 80)
            print()
            
            # Format conversation history
            history = ""
            if self.conversation_history:
                recent_history = self.conversation_history[-3:]
                history_parts = []
                for exchange in recent_history:
                    history_parts.extend([
                        f"Human: {exchange['user']}",
                        f"Assistant: {exchange['assistant']}"
                    ])
                history = "\n".join(history_parts)
            
            # Build prompt using parsed content
            prompt = self._build_multimodal_prompt(question, parsed_docs, history)
            
            # Generate response - handle streaming
            response = self.llm.invoke(prompt)
            
            # Handle streaming response
            if hasattr(response, '__iter__') and hasattr(response, 'content'):
                # Regular response object
                response_text = response.content
            elif hasattr(response, '__iter__'):
                # Streaming response - collect all chunks
                response_text = ""
                for chunk in response:
                    if hasattr(chunk, 'content') and chunk.content:
                        response_text += chunk.content
            else:
                # Fallback
                response_text = str(response)
            
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
    
    def chat_without_documents_stream(self, message: str):
        """Handle normal conversation without documents with streaming"""
        if not self.llm:
            yield "Language model not available. Please check your API configuration."
            return
        
        try:
            conversation_context = self._format_conversation_context()
            prompt = PromptTemplates.get_general_chat_prompt()
            
            # Generate streaming response with natural pacing and completion tracking
            response = ""
            buffer = ""
            word_count = 0
            last_chunk_time = time.time()
            chunks_received = 0
            
            try:
                for chunk in self.llm.stream(prompt.format(
                    conversation_context=conversation_context,
                    message=message
                )):
                    if hasattr(chunk, 'content') and chunk.content:
                        response += chunk.content
                        buffer += chunk.content
                        chunks_received += 1
                        last_chunk_time = time.time()
                        
                        # Count words to control release timing
                        word_count += len(chunk.content.split())
                        
                        # Release buffer when we have enough content (every 2-4 words)
                        if word_count >= 2 or buffer.endswith(('.', '!', '?', '\n')):
                            yield buffer
                            buffer = ""
                            word_count = 0
                    
                    # Check for timeout or stalled streaming
                    elif time.time() - last_chunk_time > 30:  # 30 second timeout
                        logger.warning("Stream timeout detected, attempting completion")
                        break
                
                # Yield any remaining content
                if buffer:
                    yield buffer
                
                # Check if response seems incomplete and try to complete it
                if self._is_response_incomplete(response) and chunks_received > 5:
                    logger.info("Response appears incomplete, attempting to complete...")
                    completion = self._attempt_completion(response, message)
                    if completion:
                        yield f"\n\n{completion}"
                        response += f"\n\n{completion}"
                        
            except Exception as stream_error:
                logger.error(f"Streaming error: {stream_error}")
                # Fallback to non-streaming if streaming fails
                try:
                    fallback_response = self.llm.invoke(prompt.format(
                        conversation_context=conversation_context,
                        message=message
                    ))
                    if hasattr(fallback_response, 'content'):
                        response = fallback_response.content
                        yield response
                except Exception as fallback_error:
                    logger.error(f"Fallback error: {fallback_error}")
                    yield "Error: Unable to generate complete response. Please try again."
                    return
            
            # Update conversation history with complete response
            self._update_conversation(message, response)
            
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            logger.error(error_msg)
            yield error_msg

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
            
            # Handle streaming response
            if hasattr(response_obj, '__iter__') and hasattr(response_obj, 'content'):
                # Regular response object
                response = response_obj.content
            elif hasattr(response_obj, '__iter__'):
                # Streaming response - collect all chunks
                response = ""
                for chunk in response_obj:
                    if hasattr(chunk, 'content') and chunk.content:
                        response += chunk.content
            else:
                # Fallback
                response = str(response_obj)
            
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
    
    def _is_response_incomplete(self, response: str) -> bool:
        """Check if response appears to be incomplete"""
        if not response:
            return True
        
        # Check for common incomplete indicators
        incomplete_indicators = [
            response.endswith('...'),
            response.endswith(' and'),
            response.endswith(' or'),
            response.endswith(' but'),
            response.endswith(' with'),
            response.endswith(' for'),
            response.endswith(' to'),
            response.endswith(' in'),
            response.endswith(' on'),
            response.endswith(' at'),
            response.endswith(' by'),
            response.endswith(' from'),
            response.endswith(','),
            response.endswith(' -'),
            len(response.strip()) < 50,  # Too short
        ]
        
        # Check if response ends mid-sentence
        stripped = response.strip()
        if stripped and not stripped[-1] in '.!?':
            return True
        
        return any(incomplete_indicators)
    
    def _attempt_completion(self, incomplete_response: str, original_question: str) -> str:
        """Attempt to complete an incomplete response"""
        try:
            completion_prompt = f"""
The following response was incomplete:
"{incomplete_response}"

Original question: "{original_question}"

Please complete this response naturally and concisely. Only provide the completion part, not the entire response again:
"""
            
            completion_response = self.llm.invoke(completion_prompt)
            if hasattr(completion_response, 'content'):
                completion = completion_response.content.strip()
                # Avoid repeating the incomplete response
                if not completion.startswith(incomplete_response[:50]):
                    return completion
            return ""
        except Exception as e:
            logger.error(f"Error in completion attempt: {e}")
            return ""

    def clear_memory(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def clear_documents(self):
        """Clear all documents from vector store and doc store"""
        try:
            # Reinitialize the FAISS manager to clear everything
            self.faiss_manager = FAISSManager(embedding_model=Config.EMBEDDING_MODEL)
            self.retriever = self.faiss_manager.retriever
            self.doc_store = {}
            logger.info("All documents cleared from RAG system")
        except Exception as e:
            logger.error(f"Error clearing documents: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        # Simple stats since FAISSManager doesn't have get_stats method
        vector_store_docs = 0
        try:
            if hasattr(self.retriever.vectorstore, 'index') and self.retriever.vectorstore.index:
                vector_store_docs = self.retriever.vectorstore.index.ntotal
        except:
            vector_store_docs = 0
        
        return {
            "documents_in_vector_store": vector_store_docs,
            "documents_in_doc_store": len(self.doc_store),
            "conversation_length": len(self.conversation_history),
            "llm_available": self.llm is not None,
            "vector_store_type": "FAISS",
            "embedding_model": getattr(self.embedding_service, 'model_name', Config.EMBEDDING_MODEL),
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