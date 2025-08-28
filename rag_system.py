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
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize LLM (using Groq for fast inference)
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="mixtral-8x7b-32768",
            temperature=0.1
        )
        
        # Initialize vector store
        self.vector_store = PineconeVectorStore(
            index=self.pc.Index(self.index_name),
            embedding=self.embeddings
        )
        
        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize conversation chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            ),
            memory=self.memory,
            verbose=True
        )
    
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
            
            # Create metadata for each chunk
            metadatas = [{
                "filename": doc["filename"],
                "chunk_id": i,
                "total_chunks": len(chunks)
            } for i in range(len(chunks))]
            
            # Add to vector store
            self.vector_store.add_texts(
                texts=chunks,
                metadatas=metadatas
            )
        
        print(f"Added {len(documents)} documents to vector store")
    
    def query(self, question: str) -> str:
        """Query the RAG system"""
        try:
            result = self.qa_chain({"question": question})
            return result["answer"]
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        
    def get_relevant_docs(self, query: str, k: int = 3):
        """Get relevant documents for a query"""
        return self.vector_store.similarity_search(query, k=k)