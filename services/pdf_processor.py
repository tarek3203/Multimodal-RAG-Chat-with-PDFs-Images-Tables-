# ===== services/pdf_processor.py =====
import fitz  # PyMuPDF
import io
from PIL import Image
from typing import Dict, List, Any
import logging

from .ocr_processor import OCRProcessor
from config import Config

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Enhanced PDF processor with OCR capabilities for Mac M1"""
    
    def __init__(self, ocr_method: str = None):
        self.ocr_method = ocr_method or Config.OCR_METHOD
        self.max_pages = Config.MAX_PDF_PAGES
        
        # Initialize OCR processor
        logger.info(f"Initializing PDF processor with OCR method: {self.ocr_method}")
        self.ocr_processor = OCRProcessor(self.ocr_method)
    
    def extract_text_basic(self, pdf_bytes: bytes) -> str:
        """Extract text using PyMuPDF basic extraction"""
        try:
            doc = fitz.open("pdf", pdf_bytes)
            text_parts = []
            
            for page_num in range(min(doc.page_count, self.max_pages)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                if page_text.strip():
                    text_parts.append(f"\\n--- Page {page_num + 1} ---\\n{page_text}")
            
            doc.close()
            return "\\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"Basic text extraction failed: {e}")
            return ""
    
    def extract_images_from_pdf(self, pdf_bytes: bytes) -> List[Dict]:
        """Extract images from PDF for OCR processing"""
        images_data = []
        
        try:
            doc = fitz.open("pdf", pdf_bytes)
            
            for page_num in range(min(doc.page_count, self.max_pages)):
                page = doc.load_page(page_num)
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Extract image
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        # Convert to PIL Image
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            image = Image.open(io.BytesIO(img_data))
                            
                            images_data.append({
                                "image": image,
                                "page": page_num + 1,
                                "image_index": img_index + 1,
                                "size": image.size
                            })
                        
                        pix = None  # Free memory
                        
                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_index} from page {page_num + 1}: {e}")
            
            doc.close()
            logger.info(f"Extracted {len(images_data)} images from PDF")
            return images_data
            
        except Exception as e:
            logger.error(f"Image extraction failed: {e}")
            return []
    
    def assess_text_quality(self, text: str) -> bool:
        """Assess if basic text extraction is sufficient"""
        if not text or len(text.strip()) < 100:
            return False
        
        # Check word density
        words = text.split()
        if len(words) < 50:
            return False
        
        # Check for reasonable character distribution
        alpha_chars = sum(1 for char in text if char.isalpha())
        if alpha_chars / len(text) < 0.3:  # Less than 30% alphabetic characters
            return False
        
        return True
    
    def ocr_images(self, images_data: List[Dict]) -> str:
        """Process images with OCR and return extracted text"""
        if not images_data:
            return ""
        
        ocr_results = []
        
        for img_data in images_data:
            try:
                # Extract text from image
                text = self.ocr_processor.extract_text(img_data["image"])
                
                if text.strip():
                    page_info = f"\\n--- Page {img_data['page']}, Image {img_data['image_index']} (OCR) ---\\n"
                    ocr_results.append(page_info + text)
                    
            except Exception as e:
                logger.warning(f"OCR failed for image on page {img_data['page']}: {e}")
        
        return "\\n".join(ocr_results)
    
    def process_pdf(self, pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Main PDF processing function with intelligent text extraction
        
        Args:
            pdf_bytes: PDF file as bytes
            filename: Name of the PDF file
            
        Returns:
            Dictionary with extracted content and metadata
        """
        logger.info(f"Processing PDF: {filename}")
        
        # Step 1: Try basic text extraction
        basic_text = self.extract_text_basic(pdf_bytes)
        
        # Step 2: Assess quality and decide if OCR is needed
        needs_ocr = not self.assess_text_quality(basic_text)
        
        extracted_content = {
            "filename": filename,
            "text_content": basic_text,
            "ocr_content": "",
            "extraction_method": "basic",
            "total_content": "",
            "metadata": {
                "page_count": 0,
                "has_images": False,
                "ocr_used": needs_ocr
            }
        }
        
        # Step 3: OCR processing if needed
        if needs_ocr:
            logger.info(f"Basic extraction insufficient for {filename}, using OCR...")
            extracted_content["extraction_method"] = "ocr"
            
            # Extract images and process with OCR
            images_data = self.extract_images_from_pdf(pdf_bytes)
            
            if images_data:
                extracted_content["metadata"]["has_images"] = True
                ocr_text = self.ocr_images(images_data)
                extracted_content["ocr_content"] = ocr_text
                logger.info(f"OCR extracted {len(ocr_text)} characters from {len(images_data)} images")
            else:
                logger.warning(f"No images found in {filename} for OCR processing")
        
        # Step 4: Combine all content
        content_parts = []
        if extracted_content["text_content"]:
            content_parts.append(extracted_content["text_content"])
        if extracted_content["ocr_content"]:
            content_parts.append(extracted_content["ocr_content"])
        
        extracted_content["total_content"] = "\\n\\n".join(content_parts)
        extracted_content["metadata"]["content_length"] = len(extracted_content["total_content"])
        
        # Page count estimation
        page_markers = extracted_content["total_content"].count("--- Page")
        extracted_content["metadata"]["page_count"] = max(1, page_markers)
        
        logger.info(f"✅ Processed {filename}: {extracted_content['metadata']['content_length']} characters, "
                   f"{extracted_content['metadata']['page_count']} pages, method: {extracted_content['extraction_method']}")
        
        return extracted_content