import os
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict

class RAGSystem:
    def __init__(self):
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "rag-chatbot-index")
        
        # Initialize Pinecone index
        self.index = self.pc.Index(self.index_name)
        
        # Initialize LLM (using Groq for fast inference)
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-70b-8192",  # Updated to a larger model
            temperature=0.1
        )
        
        # Simple conversation history (replacing deprecated memory)
        self.conversation_history = []
        self.max_history_length = 10  # Keep last 10 exchanges
    
    def add_documents(self, documents: List[Dict]):
        """Add processed documents to the vector store using Pinecone hosted embeddings"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        for doc in documents:
            # Split document into chunks
            chunks = text_splitter.split_text(doc["content"])
            
            # Prepare data for Pinecone with hosted embeddings
            records = []
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Only process non-empty chunks
                    record = {
                        "id": f"{doc['filename']}_chunk_{i}",
                        "values": chunk,  # For hosted embeddings, pass text as values
                        "metadata": {
                            "filename": doc["filename"],
                            "chunk_id": i,
                            "total_chunks": len(chunks),
                            "text": chunk
                        }
                    }
                    records.append(record)
            
            # Upsert to Pinecone in batches (hosted embeddings)
            if records:
                batch_size = 100  # Process in batches
                for i in range(0, len(records), batch_size):
                    batch = records[i:i + batch_size]
                    try:
                        self.index.upsert(vectors=batch)
                        print(f"✅ Uploaded batch {i//batch_size + 1} for {doc['filename']}")
                    except Exception as e:
                        print(f"❌ Error uploading batch for {doc['filename']}: {e}")
        
        print(f"✅ Added {len(documents)} documents to vector store")
    
    def _manage_conversation_history(self, user_message: str, ai_response: str):
        """Manage conversation history with size limits"""
        self.conversation_history.append({
            "user": user_message,
            "assistant": ai_response
        })
        
        # Keep only last max_history_length exchanges
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def _format_conversation_context(self) -> str:
        """Format conversation history for context"""
        if not self.conversation_history:
            return ""
        
        context_parts = []
        for exchange in self.conversation_history[-3:]:  # Use last 3 exchanges for context
            context_parts.append(f"Human: {exchange['user']}")
            context_parts.append(f"Assistant: {exchange['assistant']}")
        
        return "\n".join(context_parts)
    
    def query(self, question: str) -> str:
        """Query the RAG system with document context"""
        try:
            # Search for relevant documents using Pinecone's hosted embeddings
            search_results = self.index.query(
                vector=question,  # For hosted embeddings, pass text directly
                top_k=4,
                include_metadata=True,
                include_values=False
            )
            
            # Extract relevant text chunks
            context_chunks = []
            source_files = set()
            
            for match in search_results.matches:
                if hasattr(match, 'metadata') and 'text' in match.metadata:
                    context_chunks.append(match.metadata['text'])
                    if 'filename' in match.metadata:
                        source_files.add(match.metadata['filename'])
            
            # Create context from retrieved chunks
            document_context = "\n\n".join(context_chunks)
            conversation_context = self._format_conversation_context()
            
            # Generate response using LLM
            prompt = f"""You are a helpful AI assistant that answers questions based on provided documents and conversation history.

CONVERSATION HISTORY:
{conversation_context}

DOCUMENT CONTEXT:
{document_context}

CURRENT QUESTION: {question}

Instructions:
- Answer the question based primarily on the document context provided
- If the answer isn't in the documents, use your general knowledge but mention that it's not from the documents
- Be conversational and refer to previous parts of our conversation when relevant
- If you reference specific information, mention which document it's from
- Keep your response helpful and concise

Answer:"""
            
            response = self.llm.invoke(prompt)
            ai_response = response.content
            
            # Add source information if available
            if source_files:
                ai_response += f"\n\n📄 *Sources: {', '.join(source_files)}*"
            
            # Update conversation history
            self._manage_conversation_history(question, ai_response)
            
            return ai_response
            
        except Exception as e:
            error_msg = f"❌ Error processing your question: {str(e)}"
            self._manage_conversation_history(question, error_msg)
            return error_msg
    
    def chat_without_documents(self, message: str) -> str:
        """Handle normal conversation without documents"""
        try:
            conversation_context = self._format_conversation_context()
            
            prompt = f"""You are a helpful AI assistant having a natural conversation with a user.

CONVERSATION HISTORY:
{conversation_context}

CURRENT MESSAGE: {message}

Instructions:
- Respond naturally and conversationally
- Refer to previous parts of our conversation when relevant
- Be helpful, friendly, and informative
- If the user asks about uploading documents or PDFs, guide them on how to use the document upload feature

Response:"""
            
            response = self.llm.invoke(prompt)
            ai_response = response.content
            
            # Update conversation history
            self._manage_conversation_history(message, ai_response)
            
            return ai_response
            
        except Exception as e:
            error_msg = f"❌ Error processing your message: {str(e)}"
            self._manage_conversation_history(message, error_msg)
            return error_msg
    
    def clear_memory(self):
        """Clear conversation history and optionally documents"""
        self.conversation_history = []
        print("🧹 Conversation history cleared")
    
    def clear_documents(self):
        """Clear all documents from the vector store"""
        try:
            # Get all vector IDs and delete them
            # Note: This is a simple approach. For production, you might want more sophisticated document management
            self.index.delete(delete_all=True)
            print("🗑️ All documents cleared from vector store")
        except Exception as e:
            print(f"❌ Error clearing documents: {e}")
    
    def get_relevant_docs(self, query: str, k: int = 3):
        """Get relevant documents for a query (for debugging/inspection)"""
        try:
            search_results = self.index.query(
                vector=query,  # For hosted embeddings, pass text directly
                top_k=k,
                include_metadata=True,
                include_values=False
            )
            
            docs = []
            for match in search_results.matches:
                if hasattr(match, 'metadata'):
                    docs.append({
                        'content': match.metadata.get('text', ''),
                        'metadata': match.metadata,
                        'score': match.score
                    })
            return docs
            
        except Exception as e:
            print(f"❌ Error retrieving documents: {e}")
            return []
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the current conversation"""
        if not self.conversation_history:
            return "No conversation yet."
        
        return f"Conversation has {len(self.conversation_history)} exchanges. Topics discussed include the recent questions and responses."