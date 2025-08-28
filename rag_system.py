import os
from pinecone import Pinecone
try:
    from langchain_pinecone import PineconeVectorStore
except ImportError:
    from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from typing import List, Dict

class RAGSystem:
    def __init__(self):
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "rag-chatbot-index")
        
        # Initialize Pinecone index for inference (using hosted embeddings)
        self.index = self.pc.Index(self.index_name)
        
        # Initialize LLM (using Groq for fast inference)
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="mixtral-8x7b-32768",
            temperature=0.1
        )
        
        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize conversation chain - will be set up after adding documents
        self.qa_chain = None
    
    def add_documents(self, documents: List[Dict]):
        """Add processed documents to the vector store"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        for doc in documents:
            # Split document into chunks
            chunks = text_splitter.split_text(doc["content"])
            
            # Prepare data for Pinecone with hosted embeddings
            vectors_to_upsert = []
            for i, chunk in enumerate(chunks):
                vector_id = f"{doc['filename']}_chunk_{i}"
                metadata = {
                    "filename": doc["filename"],
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "text": chunk  # This field will be embedded automatically
                }
                vectors_to_upsert.append({
                    "id": vector_id,
                    "metadata": metadata
                })
            
            # Upsert to Pinecone (embeddings will be generated automatically)
            self.index.upsert(vectors=vectors_to_upsert)
        
        print(f"Added {len(documents)} documents to vector store")
        
        # Set up QA chain after documents are added
        self._setup_qa_chain()
    
    def _setup_qa_chain(self):
        """Set up the QA chain with a simple retriever"""
        # For now, we'll use a simple approach without langchain retriever
        # since we're using Pinecone's hosted embeddings
        pass
    
    def query(self, question: str) -> str:
        """Query the RAG system"""
        try:
            # Search for relevant documents using Pinecone's hosted embeddings
            search_results = self.index.query(
                vector=None,  # Pinecone will embed the query text automatically
                top_k=4,
                include_metadata=True,
                filter=None,
                query_text=question  # This will be embedded automatically
            )
            
            # Extract relevant text chunks
            context_chunks = []
            for match in search_results.matches:
                if hasattr(match, 'metadata') and 'text' in match.metadata:
                    context_chunks.append(match.metadata['text'])
            
            # Create context from retrieved chunks
            context = "\n\n".join(context_chunks)
            
            # Generate response using LLM
            prompt = f"""Based on the following context, answer the question. If the answer cannot be found in the context, say so.

Context:
{context}

Question: {question}

Answer:"""
            
            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        
    def get_relevant_docs(self, query: str, k: int = 3):
        """Get relevant documents for a query"""
        try:
            search_results = self.index.query(
                vector=None,
                top_k=k,
                include_metadata=True,
                query_text=query
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
            print(f"Error retrieving documents: {e}")
            return []