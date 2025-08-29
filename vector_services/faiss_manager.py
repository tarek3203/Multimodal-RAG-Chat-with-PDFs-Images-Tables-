# ===== vector_services/faiss_manager.py (Updated for better Mac M1 performance) =====
from langchain_community.vectorstores import FAISS
import os
import logging
import shutil
from typing import List, Dict, Any, Optional
import warnings
import pickle

logger = logging.getLogger('VectorService')

class FAISSManager:
    """FAISS vector database manager optimized for Mac M1 with local storage."""
    
    def __init__(self, 
                 persist_directory=None, 
                 index_name="default", 
                 embedding_service=None):
        """
        Initialize FAISS vector store manager.
        
        Args:
            persist_directory: Directory to store FAISS indexes
            index_name: Name of the index to use
            embedding_service: EmbeddingService instance, creates one if None
        """
        # Import here to avoid circular import
        if embedding_service is None:
            from .embeddings import EmbeddingService
            embedding_service = EmbeddingService()
            
        self.persist_directory = persist_directory or "./vector_storage"
        self.index_name = index_name
        self.index_path = os.path.join(self.persist_directory, self.index_name)
        
        self.embedding_service = embedding_service
        
        # Suppress FAISS warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="faiss")
        warnings.filterwarnings("ignore", message=".*Failed to load GPU Faiss.*")
        
        # Create directory
        os.makedirs(self.persist_directory, exist_ok=True)
        logger.info(f"FAISS manager initialized: {self.index_path}")
        
        # Load or create index
        self.index = self._load_or_create_index()
        
        # Track document metadata separately for better management
        self.metadata_path = os.path.join(self.index_path, "metadata.pkl")
        self.document_metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load document metadata"""
        try:
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")
        
        return {"documents": [], "total_chunks": 0}
    
    def _save_metadata(self):
        """Save document metadata"""
        try:
            os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.document_metadata, f)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _load_or_create_index(self):
        """Load existing index or return None for lazy creation."""
        try:
            if os.path.exists(self.index_path) and os.path.isdir(self.index_path):
                # Check if index files exist
                index_file = os.path.join(self.index_path, "index.faiss")
                if os.path.exists(index_file):
                    logger.info(f"Loading existing FAISS index from {self.index_path}")
                    return FAISS.load_local(
                        self.index_path, 
                        self.embedding_service.model,
                        allow_dangerous_deserialization=True
                    )
            
            logger.info("No existing index found, will create when documents are added")
            return None
                
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            logger.info("Will create new index when documents are added")
            return None
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        """
        Add text documents to the vector store.
        
        Args:
            texts: List of text documents to add
            metadatas: Optional list of metadata dictionaries, one per document
            
        Returns:
            List of document IDs
        """
        if not texts:
            logger.warning("No texts provided to add to vector store")
            return []
        
        # Filter empty texts
        valid_items = []
        for i, text in enumerate(texts):
            if text and text.strip():
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                valid_items.append((text.strip(), metadata))
        
        if not valid_items:
            logger.warning("No valid texts after filtering")
            return []
        
        valid_texts, valid_metadatas = zip(*valid_items)
        
        try:
            if self.index is None:
                # Create new index
                logger.info(f"Creating new FAISS index with {len(valid_texts)} documents")
                self.index = FAISS.from_texts(
                    texts=list(valid_texts),
                    embedding=self.embedding_service.model,
                    metadatas=list(valid_metadatas)
                )
                
                # Update metadata tracking
                filenames = [meta.get('filename', 'unknown') for meta in valid_metadatas]
                self.document_metadata["documents"].extend(set(filenames))
                self.document_metadata["total_chunks"] += len(valid_texts)
                
            else:
                # Add to existing index
                logger.info(f"Adding {len(valid_texts)} documents to existing FAISS index")
                self.index.add_texts(texts=list(valid_texts), metadatas=list(valid_metadatas))
                
                # Update metadata tracking
                filenames = [meta.get('filename', 'unknown') for meta in valid_metadatas]
                for filename in set(filenames):
                    if filename not in self.document_metadata["documents"]:
                        self.document_metadata["documents"].append(filename)
                self.document_metadata["total_chunks"] += len(valid_texts)
            
            # Save index and metadata
            self._save_index()
            self._save_metadata()
            
            logger.info(f"✅ Successfully added {len(valid_texts)} chunks to vector store")
            return [f"doc_{i}" for i in range(len(valid_texts))]
                
        except Exception as e:
            logger.error(f"Failed to add texts to vector store: {str(e)}")
            raise
    
    def _save_index(self):
        """Save the current index to disk."""
        if self.index:
            try:
                logger.info(f"Saving FAISS index to {self.index_path}")
                self.index.save_local(self.index_path)
            except Exception as e:
                logger.error(f"Failed to save FAISS index: {str(e)}")
                raise
    
    def similarity_search(self, query: str, k: int = 4):
        """
        Search for documents similar to the query.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of results with document content and metadata
        """
        if not self.index:
            logger.warning("No index available for similarity search")
            return []
            
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []
            
        try:
            # Perform similarity search with scores
            results = self.index.similarity_search_with_score(query.strip(), k=k)
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })
            
            logger.debug(f"Found {len(formatted_results)} similar documents for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {str(e)}")
            return []
    
    def delete_index(self):
        """Delete the current index from disk."""
        try:
            if os.path.exists(self.index_path):
                logger.info(f"Deleting FAISS index at {self.index_path}")
                shutil.rmtree(self.index_path)
                
            self.index = None
            self.document_metadata = {"documents": [], "total_chunks": 0}
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete FAISS index: {str(e)}")
            raise
    
    def get_stats(self):
        """Get statistics about the vector store."""
        doc_count = 0
        
        if self.index and hasattr(self.index, 'docstore'):
            try:
                doc_count = len(self.index.docstore._dict)
            except:
                # Fallback to metadata
                doc_count = self.document_metadata.get("total_chunks", 0)
        
        return {
            "documents": doc_count,
            "unique_files": len(self.document_metadata.get("documents", [])),
            "index_name": self.index_name,
            "index_path": self.index_path,
            "embedding_model": self.embedding_service.model_name if self.embedding_service else None,
            "index_exists": self.index is not None
        }
    
    def get_document_list(self) -> List[str]:
        """Get list of processed document filenames"""
        return self.document_metadata.get("documents", [])