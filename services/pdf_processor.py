# ===== services/pdf_processor.py =====
"""
Multimodal PDF processor using unstructured library for comprehensive extraction
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import tempfile
import base64
from config import Config
from prompts import PromptTemplates

logger = logging.getLogger(__name__)

class MultimodalPDFProcessor:
    """PDF processor with multimodal capabilities using unstructured library"""
    
    def __init__(self):
        self.max_pages = Config.MAX_PDF_PAGES
        self.temp_dir = None
        
        # Initialize summarization chains
        self._init_summarization_chains()
        
        logger.info("Multimodal PDF processor initialized")
    
    def _init_summarization_chains(self):
        """Initialize the summarization chains for different content types"""
        try:
            from langchain_groq import ChatGroq
            from langchain_core.output_parsers import StrOutputParser
            
            if Config.GROQ_API_KEY:
                # Initialize Groq model
                self.text_model = ChatGroq(
                    api_key=Config.GROQ_API_KEY,
                    model=Config.GROQ_MODEL,
                    temperature=0.3
                )
                
                # Text summarization chain
                text_prompt = PromptTemplates.get_text_summary_prompt()
                self.text_summarize_chain = {"element": lambda x: x} | text_prompt | self.text_model | StrOutputParser()
                
                # Table summarization chain  
                table_prompt = PromptTemplates.get_table_summary_prompt()
                self.table_summarize_chain = {"element": lambda x: x} | table_prompt | self.text_model | StrOutputParser()
                
                logger.info("Text and Table summarization chains initialized with Groq")
            else:
                self.text_model = None
                self.text_summarize_chain = None
                self.table_summarize_chain = None
                logger.warning("Groq API key not found - text/table summarization disabled")
            
            # Image analysis chain (separate from text/table)
            self.image_model = None
            self.image_analyze_chain = None
            
            # Try Google Gemini first for image analysis
            if Config.GOOGLE_API_KEY:
                try:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    
                    self.image_model = ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash",
                        api_key=Config.GOOGLE_API_KEY,
                        temperature=0.3,
                        max_output_tokens=300
                    )
                    
                    # Use prompt from prompts file - clean and structured
                    image_prompt = PromptTemplates.get_image_analysis_prompt()
                    self.image_analyze_chain = image_prompt | self.image_model | StrOutputParser()
                    logger.info("Image analysis chain initialized with Gemini")
                    
                except ImportError:
                    logger.warning("langchain-google-genai not available")
            
            # Fallback to OpenAI if Gemini not available
            if not self.image_model and Config.OPENAI_API_KEY:
                try:
                    from langchain_openai import ChatOpenAI
                    
                    self.image_model = ChatOpenAI(
                        model="gpt-4o",
                        api_key=Config.OPENAI_API_KEY,
                        temperature=0.3,
                        max_tokens=300
                    )
                    
                    # Use the same prompt from prompts file for consistency
                    image_prompt = PromptTemplates.get_image_analysis_prompt()
                    self.image_analyze_chain = image_prompt | self.image_model | StrOutputParser()
                    logger.info("Image analysis chain initialized with OpenAI")
                    
                except ImportError:
                    logger.warning("langchain-openai not available")
            
            if not self.image_model:
                logger.warning("No vision model available - image analysis disabled")
                
        except Exception as e:
            logger.error(f"Error initializing summarization chains: {e}")
            self.text_summarize_chain = None
            self.table_summarize_chain = None
            self.image_analyze_chain = None
    
    def _partition_pdf(self, pdf_path: str) -> List[Any]:
        """Partition PDF using unstructured library"""
        try:
            from unstructured.partition.pdf import partition_pdf
            
            # Create temporary directory for image extraction
            self.temp_dir = tempfile.mkdtemp()
            
            logger.info(f"Partitioning PDF: {pdf_path}")
            
            chunks = partition_pdf(
                filename=pdf_path,
                infer_table_structure=True,            # Extract tables
                strategy="hi_res",                     # High resolution for better quality
                extract_image_block_types=["Image"],   # Extract images
                extract_image_block_to_payload=True,   # Get base64 encoded images
                chunking_strategy="by_title",          # Chunk by document structure
                max_characters=10000,                  # Max characters per chunk
                combine_text_under_n_chars=2000,       # Combine small chunks
                new_after_n_chars=6000,               # New chunk after this many chars
            )
            
            logger.info(f"Successfully partitioned PDF into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error partitioning PDF: {e}")
            return []
    
    def get_tables(self, chunks):
        """Extract all Table elements from chunks and from within CompositeElement chunks"""
        tables = []

        # First, look for direct Table type chunks
        for chunk in chunks:
            if "Table" in str(type(chunk)):
                tables.append(chunk)

        # Then, look for tables within CompositeElement chunks
        for chunk in chunks:
            if "CompositeElement" in str(type(chunk)):
                if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
                    chunk_els = chunk.metadata.orig_elements
                    for el in chunk_els:
                        if "Table" in str(type(el)):
                            tables.append(el)

        logger.info(f"Extracted {len(tables)} table elements")
        return tables

    def get_texts(self, chunks):
        """Extract all text-based elements from chunks (excluding tables and images)"""
        texts = []
        excluded_types = ["Table", "Image"]

        # First, look for direct text-based chunks (excluding tables, images, and CompositeElement)
        for chunk in chunks:
            chunk_type = str(type(chunk))
            if not any(excluded_type in chunk_type for excluded_type in excluded_types) and "CompositeElement" not in chunk_type:
                texts.append(chunk)

        # Then, look for text elements within CompositeElement chunks (excluding tables and images)
        for chunk in chunks:
            if "CompositeElement" in str(type(chunk)):
                if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
                    chunk_els = chunk.metadata.orig_elements
                    for el in chunk_els:
                        el_type = str(type(el))
                        if not any(excluded_type in el_type for excluded_type in excluded_types):
                            texts.append(el)

        logger.info(f"Extracted {len(texts)} text elements")
        return texts

    def get_images(self, chunks):
        """Extract all Image elements and images within other chunks"""
        images_b64 = []

        # First, look for direct Image type chunks
        for chunk in chunks:
            if "Image" in str(type(chunk)):
                if hasattr(chunk.metadata, 'image_base64'):
                    images_b64.append(chunk.metadata.image_base64)

        # Then, look for images within other chunk types (like CompositeElement)
        for chunk in chunks:
            if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
                chunk_els = chunk.metadata.orig_elements
                for el in chunk_els:
                    if "Image" in str(type(el)):
                        if hasattr(el.metadata, 'image_base64'):
                            images_b64.append(el.metadata.image_base64)

        logger.info(f"Extracted {len(images_b64)} image elements")
        return images_b64
    
    
    def _summarize_elements(self, texts: List[Any], tables: List[Any], images: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """Generate summaries for all extracted elements"""
        text_summaries = []
        table_summaries = []
        image_summaries = []
        
        # Summarize texts
        if texts and self.text_summarize_chain:
            try:
                logger.info(f"Summarizing {len(texts)} text elements")
                text_content = [str(text) for text in texts]
                text_summaries = self.text_summarize_chain.batch(text_content, {"max_concurrency": 3})
                logger.info(f"Generated {len(text_summaries)} text summaries")
            except Exception as e:
                logger.error(f"Error summarizing texts: {e}")
                # Fallback to original content
                text_summaries = [str(text) for text in texts]
        else:
            text_summaries = [str(text) for text in texts]
        
        # Summarize tables
        if tables and self.text_summarize_chain:
            try:
                logger.info(f"Summarizing {len(tables)} table elements")
                # Extract HTML representation of tables
                table_content = []
                for table in tables:
                    if hasattr(table.metadata, 'text_as_html'):
                        table_content.append(table.metadata.text_as_html)
                    else:
                        table_content.append(str(table))
                
                table_summaries = self.text_summarize_chain.batch(table_content, {"max_concurrency": 3})
                logger.info(f"Generated {len(table_summaries)} table summaries")
            except Exception as e:
                logger.error(f"Error summarizing tables: {e}")
                # Fallback to original content
                table_summaries = [str(table) for table in tables]
        else:
            table_summaries = [str(table) for table in tables]
        
        # Analyze images
        if images and self.image_analyze_chain:
            try:
                logger.info(f"Analyzing {len(images)} image elements")
                # Process images in batches to avoid rate limits
                image_summaries = self.image_analyze_chain.batch(images, {"max_concurrency": 2})
                logger.info(f"Generated {len(image_summaries)} image descriptions")
            except Exception as e:
                logger.error(f"Error analyzing images: {e}")
                # Fallback to placeholder descriptions
                image_summaries = [f"Image {i+1} from PDF document" for i in range(len(images))]
        else:
            image_summaries = [f"Image {i+1} from PDF document" for i in range(len(images))]
        
        return text_summaries, table_summaries, image_summaries
    
    def process_pdf(self, pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Main PDF processing function"""
        logger.info(f"Processing multimodal PDF: {filename}")
        
        try:
            # Save bytes to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(pdf_bytes)
                temp_pdf_path = temp_file.name
            
            # Partition PDF
            chunks = self._partition_pdf(temp_pdf_path)
            
            if not chunks:
                logger.warning(f"No content extracted from {filename}")
                return {
                    "filename": filename,
                    "content": "No content could be extracted from this PDF",
                    "texts": [],
                    "tables": [],
                    "images": [],
                    "text_summaries": [],
                    "table_summaries": [],
                    "image_summaries": [],
                    "metadata": {"extraction_method": "multimodal_failed"}
                }
            
            # texts extractions
            texts = self.get_texts(chunks)
            
            # table extractions
            tables = self.get_tables(chunks)

            # image extractions
            images = self.get_images(chunks)

            # Generate summaries
            text_summaries, table_summaries, image_summaries = self._summarize_elements(texts, tables, images)
            
            
            # Cleanup temporary files
            try:
                Path(temp_pdf_path).unlink()
                if self.temp_dir:
                    import shutil
                    shutil.rmtree(self.temp_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Error cleaning up temporary files: {e}")
            
            result = {
                "filename": filename,
                # "content": total_content,  # Combined content for compatibility
                "texts": texts,
                "tables": tables, 
                "images": images,
                "text_summaries": text_summaries,
                "table_summaries": table_summaries,
                "image_summaries": image_summaries,
                "metadata": {
                    "extraction_method": "multimodal_unstructured",
                    "page_count": len(chunks),
                    # "content_length": len(total_content),
                    "text_count": len(texts),
                    "table_count": len(tables),
                    "image_count": len(images),
                    "has_vision_analysis": self.image_analyze_chain is not None
                }
            }
            
            logger.info(f"Successfully processed {filename}: "
                       f"{len(text_summaries)} texts, {len(table_summaries)} tables, {len(image_summaries)} images")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF {filename}: {e}")
            return {
                "filename": filename,
                "content": f"Error processing PDF: {str(e)}",
                "texts": [],
                "tables": [],
                "images": [],
                "text_summaries": [],
                "table_summaries": [],
                "image_summaries": [],
                "metadata": {"extraction_method": "error", "error": str(e)}
            }


# Legacy compatibility - alias for existing code
PDFProcessor = MultimodalPDFProcessor