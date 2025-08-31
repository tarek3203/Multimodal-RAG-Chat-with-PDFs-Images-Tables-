# ===== vector_services/embeddings.py =====
"""
Simple embedding service - just a wrapper around HuggingFace embeddings
"""
import os
import logging
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger('VectorService')

def get_embeddings(model_name: str = None):
    """
    Get HuggingFace embeddings model
    
    Args:
        model_name: Name of the HuggingFace model
        
    Returns:
        HuggingFaceEmbeddings instance
    """
    model_name = model_name or os.environ.get(
        "EMBEDDING_MODEL", 
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    
    logger.info(f"Loading embedding model: {model_name}")
    
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu', 'trust_remote_code': True},
        encode_kwargs={'normalize_embeddings': True}
    )