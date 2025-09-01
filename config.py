# ===== config.py =====
"""
Enhanced configuration settings for the multimodal RAG system
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set environment variable to avoid tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Config:
    """Configuration settings for the multimodal OCR RAG system"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent
    VECTOR_FOLDER = PROJECT_ROOT / "vector_storage"
    MODELS_FOLDER = PROJECT_ROOT / "models"
    
    # PDF Processing settings
    MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "20"))
    
    # API Keys for different services
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # For Gemini vision
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Fallback for vision
    
    # LLM Configuration
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  # For text generation
    
    # Embedding Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Vector Store Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # FAISS Configuration
    FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "multimodal_documents")
    
    # Vision Model Priority (which to try first for image analysis)
    VISION_MODEL_PRIORITY = os.getenv("VISION_MODEL_PRIORITY", "gemini,openai").split(",")
    
    # Processing limits
    MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "3"))
    IMAGE_ANALYSIS_TIMEOUT = int(os.getenv("IMAGE_ANALYSIS_TIMEOUT", "30"))
    
    # Legacy OCR settings (kept for backward compatibility)
    OCR_METHOD = os.getenv("OCR_METHOD", "unstructured")
    LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", str(MODELS_FOLDER / "mistral-7b-instruct-v0.1.Q4_0.gguf"))
    LLM_CONTEXT_SIZE = int(os.getenv("LLM_CONTEXT_SIZE", "4096"))
    LLM_THREADS = int(os.getenv("LLM_THREADS", "4"))
    
    @classmethod
    def ensure_directories(cls):
        """Ensure required directories exist"""
        cls.VECTOR_FOLDER.mkdir(parents=True, exist_ok=True)
        cls.MODELS_FOLDER.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_available_apis(cls) -> dict:
        """Get status of available API keys"""
        return {
            "groq": bool(cls.GROQ_API_KEY),
            "google": bool(cls.GOOGLE_API_KEY),
            "openai": bool(cls.OPENAI_API_KEY)
        }
    
    @classmethod
    def get_vision_model_config(cls) -> dict:
        """Get vision model configuration based on available APIs"""
        config = {"available": False, "model": None, "api": None}
        
        for model_type in cls.VISION_MODEL_PRIORITY:
            model_type = model_type.strip().lower()
            
            if model_type == "gemini" and cls.GOOGLE_API_KEY:
                config = {
                    "available": True,
                    "model": "gemini-1.5-flash",
                    "api": "google",
                    "cost_per_1k_tokens": 0.075  # Much cheaper than OpenAI
                }
                break
            elif model_type == "openai" and cls.OPENAI_API_KEY:
                config = {
                    "available": True,
                    "model": "gpt-4o",
                    "api": "openai", 
                    "cost_per_1k_tokens": 2.50  # More expensive but very capable
                }
                break
        
        return config
    
    @classmethod
    def validate_configuration(cls) -> dict:
        """Validate current configuration and return status"""
        status = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Check essential API keys
        if not cls.GROQ_API_KEY:
            status["errors"].append("GROQ_API_KEY is required for text generation")
            status["valid"] = False
        
        # Check vision capabilities
        vision_config = cls.get_vision_model_config()
        if not vision_config["available"]:
            status["warnings"].append(
                "No vision API available (GOOGLE_API_KEY or OPENAI_API_KEY). "
                "Image analysis will be limited."
            )
        
        # Check directories
        try:
            cls.ensure_directories()
        except Exception as e:
            status["errors"].append(f"Cannot create required directories: {e}")
            status["valid"] = False
        
        return status