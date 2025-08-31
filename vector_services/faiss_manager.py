# ===== vector_services/faiss_manager.py =====
"""
Simple FAISS manager - just initializes vectorstore and docstore
"""
import uuid
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever

class FAISSManager:
    """FAISS manager for handling vector storage and retrieval"""

    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Initialize HuggingFace embeddings (like OpenAIEmbeddings in colab)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize vectorstore with a dummy document to start
        self.vectorstore = FAISS.from_texts(["init"], embedding=self.embeddings)
        
        # The storage layer for the parent documents 
        self.docstore = InMemoryStore()
        
        # Document ID key
        self.id_key = "doc_id"
        
        # The retriever (empty to start) - exactly like colab notebook
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            id_key=self.id_key,
        )