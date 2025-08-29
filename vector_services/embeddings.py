# ===== vector_services/embeddings.py (Updated for Mac M1) =====
from langchain_huggingface import HuggingFaceEmbeddings
import os
import logging
import torch

logger = logging.getLogger('VectorService')

class EmbeddingService:
    """Service for generating embeddings using HuggingFace models - optimized for Mac M1."""
    
    def __init__(self, model_name=None):
        """
        Initialize the embedding service with a HuggingFace model.
        
        Args:
            model_name: Name of the HuggingFace model to use for embeddings.
                        Defaults to environment variable or a sensible default.
        """
        self.model_name = model_name or os.environ.get(
            "EMBEDDING_MODEL", 
            "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight model for Mac M1
        )
        
        logger.info(f"Initializing embedding service with model: {self.model_name}")
        
        try:
            # Configure for Mac M1 - force CPU usage
            model_kwargs = {
                'device': 'cpu',  # Force CPU for M1 compatibility
                'trust_remote_code': True
            }
            
            # Check if MPS is available (Apple Silicon GPU) but prefer CPU for stability
            if torch.backends.mps.is_available():
                logger.info("MPS (Apple Silicon GPU) available, but using CPU for stability")
            
            self.model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs=model_kwargs,
                encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity
            )
            
            logger.info(f"Embedding model '{self.model_name}' loaded successfully on CPU")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise RuntimeError(f"Embedding model initialization failed: {str(e)}")
    
    def embed_documents(self, texts):
        """
        Generate embeddings for a list of text documents.
        
        Args:
            texts: List of strings to embed
            
        Returns:
            List of embeddings (vectors)
        """
        if not texts:
            logger.warning("No texts provided for embedding")
            return []
            
        try:
            # Filter out empty texts
            valid_texts = [text for text in texts if text.strip()]
            if not valid_texts:
                return []
            
            logger.info(f"Generating embeddings for {len(valid_texts)} documents")
            embeddings = self.model.embed_documents(valid_texts)
            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise
    
    def embed_query(self, text):
        """
        Generate embedding for a single query text.
        
        Args:
            text: String to embed
            
        Returns:
            Embedding vector
        """
        if not text or not text.strip():
            logger.warning("Empty query text provided")
            return None
            
        try:
            logger.debug(f"Generating query embedding for: {text[:50]}...")
            embedding = self.model.embed_query(text.strip())
            return embedding
            
        except Exception as e:
            logger.error(f"Query embedding failed: {str(e)}")
            raise
    
    def get_embedding_dimension(self):
        """Get the dimension of the embeddings"""
        try:
            # Test with a dummy text to get dimensions
            test_embedding = self.embed_query("test")
            return len(test_embedding) if test_embedding else None
        except:
            # Default dimension for all-MiniLM-L6-v2
            return 384