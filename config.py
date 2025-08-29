# ===== config.py =====
import os
from pathlib import Path

class Config:
    """Configuration settings for the OCR RAG system"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent
    VECTOR_FOLDER = PROJECT_ROOT / "vector_storage"
    MODELS_FOLDER = PROJECT_ROOT / "models"
    
    # OCR settings
    OCR_METHOD = os.getenv("OCR_METHOD", "trocr")  # trocr, easyocr, got_ocr
    MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "15"))
    
    # Local LLM settings
    # Ollama settings
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2:7b-chat")
    LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", str(MODELS_FOLDER / "mistral-7b-instruct-v0.1.Q4_0.gguf"))
    LLM_CONTEXT_SIZE = int(os.getenv("LLM_CONTEXT_SIZE", "2048"))
    LLM_THREADS = int(os.getenv("LLM_THREADS", "4"))
    
    # Embedding settings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Chunk settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    @classmethod
    def ensure_directories(cls):
        """Ensure required directories exist"""
        cls.VECTOR_FOLDER.mkdir(exist_ok=True)
        cls.MODELS_FOLDER.mkdir(exist_ok=True)