# ===== services/enhanced_pdf_processor.py =====
# Enhanced PDF processor with systematic extraction: text → tables → images
import fitz  # PyMuPDF
import pdfplumber
import io
from PIL import Image
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
import cv2
import numpy as np

from config import Config

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Enhanced PDF processor with systematic text, table, and image extraction"""
    
    def __init__(self, ocr_method: str = None):
        self.ocr_method = ocr_method or Config.OCR_METHOD
        self.max_pages = Config.MAX_PDF_PAGES
        
        # Initialize OCR processor for images
        self._init_image_ocr()
        
        logger.info(f"Enhanced PDF processor initialized with OCR: {self.ocr_method}")
    
    def _init_image_ocr(self):
        """Initialize the best OCR method for image text extraction"""
        try:
            # Try PaddleOCR first (best for structured text)
            import paddleocr
            self.image_ocr = paddleocr.PaddleOCR(
                use_angle_cls=True, 
                lang='en',
                # show_log=False,
                # use_gpu=False  # CPU for M1 compatibility
            )
            self.ocr_method = "paddleocr"
            logger.info("Using PaddleOCR for image text extraction")
            
        except ImportError:
            # Fallback to EasyOCR
            try:
                import easyocr
                self.image_ocr = easyocr.Reader(['en'], gpu=False)
                self.ocr_method = "easyocr"
                logger.info("Using EasyOCR for image text extraction")
                
            except ImportError:
                logger.warning("No OCR library available for images. Install paddleocr or easyocr")
                self.image_ocr = None
    
    def extract_text_from_page(self, page) -> str:
        """Extract basic text from a single page"""
        try:
            text = page.get_text()
            return text.strip() if text else ""
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return ""
    
    def detect_and_extract_tables(self, pdf_bytes: bytes, page_num: int) -> List[Dict]:
        """Detect and extract tables from a specific page with structural formatting"""
        tables_data = []
        
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                if page_num < len(pdf.pages):
                    page = pdf.pages[page_num]
                    
                    # Extract tables from the page
                    tables = page.extract_tables()
                    
                    for table_idx, table in enumerate(tables):
                        if table and len(table) > 1:  # Must have headers and data
                            # Convert to structured format
                            structured_table = self._format_table_structurally(
                                table, page_num + 1, table_idx + 1
                            )
                            tables_data.append(structured_table)
                            
        except Exception as e:
            logger.error(f"Table extraction failed for page {page_num + 1}: {e}")
        
        return tables_data
    
    def _format_table_structurally(self, table: List[List], page_num: int, table_num: int) -> Dict:
        """Format table data in a structured, readable format"""
        if not table:
            return {}
        
        # Clean and prepare table data
        cleaned_table = []
        for row in table:
            cleaned_row = []
            for cell in row:
                # Clean cell content
                cell_content = str(cell).strip() if cell is not None else ""
                cleaned_row.append(cell_content)
            cleaned_table.append(cleaned_row)
        
        # Convert to pandas DataFrame for better handling
        try:
            df = pd.DataFrame(cleaned_table[1:], columns=cleaned_table[0])
            
            # Create markdown-style table
            markdown_table = self._create_markdown_table(df)
            
            # Also create structured dictionary
            structured_data = {
                "headers": cleaned_table[0],
                "rows": cleaned_table[1:],
                "row_count": len(cleaned_table) - 1,
                "col_count": len(cleaned_table[0]) if cleaned_table else 0
            }
            
            return {
                "page": page_num,
                "table_number": table_num,
                "markdown_format": markdown_table,
                "structured_data": structured_data,
                "source_type": "table_extraction"
            }
            
        except Exception as e:
            logger.error(f"Table formatting failed: {e}")
            return {
                "page": page_num,
                "table_number": table_num,
                "raw_data": cleaned_table,
                "source_type": "table_extraction"
            }
    
    def _create_markdown_table(self, df: pd.DataFrame) -> str:
        """Create a well-formatted markdown table"""
        if df.empty:
            return ""
        
        # Create header row
        headers = "| " + " | ".join(str(col) for col in df.columns) + " |"
        separator = "|" + "|".join("---" for _ in df.columns) + "|"
        
        # Create data rows
        rows = []
        for _, row in df.iterrows():
            row_str = "| " + " | ".join(str(cell) for cell in row) + " |"
            rows.append(row_str)
        
        return "\n".join([headers, separator] + rows)
    
    def extract_images_with_text(self, pdf_bytes: bytes, page_num: int) -> List[Dict]:
        """Extract images and their text content with structure preservation"""
        images_with_text = []
        
        try:
            doc = fitz.open("pdf", pdf_bytes)
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
                        
                        # Extract text from image with structure
                        extracted_text = self._extract_structured_text_from_image(image)
                        
                        if extracted_text:
                            images_with_text.append({
                                "page": page_num + 1,
                                "image_index": img_index + 1,
                                "extracted_text": extracted_text,
                                "image_size": image.size,
                                "source_type": "image_ocr"
                            })
                    
                    pix = None  # Free memory
                    
                except Exception as e:
                    logger.warning(f"Image processing failed for image {img_index} on page {page_num + 1}: {e}")
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Image extraction failed for page {page_num + 1}: {e}")
        
        return images_with_text
    
    def _extract_structured_text_from_image(self, image: Image.Image) -> str:
        """Extract text from image while preserving structure"""
        if not self.image_ocr:
            return ""
        
        try:
            # Convert PIL to numpy for OCR processing
            img_array = np.array(image)
            
            if self.ocr_method == "paddleocr":
                # PaddleOCR preserves text positioning
                result = self.image_ocr.ocr(img_array, cls=True)
                
                if result and result[0]:
                    # Sort by vertical position to maintain reading order
                    sorted_results = sorted(result[0], key=lambda x: (x[0][0][1], x[0][0][0]))
                    
                    text_blocks = []
                    current_line = []
                    current_y = None
                    line_threshold = 10  # pixels
                    
                    for detection in sorted_results:
                        bbox, (text, confidence) = detection
                        y_pos = bbox[0][1]  # Top-left y coordinate
                        
                        if confidence > 0.7:  # Only high-confidence text
                            if current_y is None or abs(y_pos - current_y) < line_threshold:
                                # Same line
                                current_line.append(text)
                                current_y = y_pos
                            else:
                                # New line
                                if current_line:
                                    text_blocks.append(" ".join(current_line))
                                current_line = [text]
                                current_y = y_pos
                    
                    # Add the last line
                    if current_line:
                        text_blocks.append(" ".join(current_line))
                    
                    return "\n".join(text_blocks)
            
            elif self.ocr_method == "easyocr":
                # EasyOCR processing
                results = self.image_ocr.readtext(img_array)
                
                # Sort by position to maintain structure
                sorted_results = sorted(results, key=lambda x: (x[0][0][1], x[0][0][0]))
                
                text_parts = []
                for bbox, text, confidence in sorted_results:
                    if confidence > 0.7:  # Only high-confidence text
                        text_parts.append(text.strip())
                
                return "\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"OCR text extraction failed: {e}")
        
        return ""
    
    def process_single_page(self, pdf_bytes: bytes, page_num: int) -> Dict[str, Any]:
        """Process a single page systematically: text → tables → images"""
        page_content = {
            "page_number": page_num + 1,
            "text_content": "",
            "tables": [],
            "images_text": [],
            "extraction_methods": []
        }
        
        try:
            # Open page with PyMuPDF for basic text extraction
            doc = fitz.open("pdf", pdf_bytes)
            page = doc.load_page(page_num)
            
            # Step 1: Extract basic text
            text_content = self.extract_text_from_page(page)
            
            if text_content and len(text_content.strip()) > 50:
                page_content["text_content"] = text_content
                page_content["extraction_methods"].append("basic_text")
            
            doc.close()
            
            # Step 2: Detect and extract tables
            tables = self.detect_and_extract_tables(pdf_bytes, page_num)
            if tables:
                page_content["tables"] = tables
                page_content["extraction_methods"].append("table_extraction")
            
            # Step 3: Extract images with text if no good text/tables found
            # or if page seems to have embedded images
            should_extract_images = (
                len(page_content["text_content"]) < 100 or
                not page_content["tables"] or
                self._page_has_significant_images(pdf_bytes, page_num)
            )
            
            if should_extract_images:
                images_text = self.extract_images_with_text(pdf_bytes, page_num)
                if images_text:
                    page_content["images_text"] = images_text
                    page_content["extraction_methods"].append("image_ocr")
        
        except Exception as e:
            logger.error(f"Page processing failed for page {page_num + 1}: {e}")
        
        return page_content
    
    def _page_has_significant_images(self, pdf_bytes: bytes, page_num: int) -> bool:
        """Check if page has significant images worth processing"""
        try:
            doc = fitz.open("pdf", pdf_bytes)
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)
            doc.close()
            
            # Consider significant if more than 2 images or large images
            return len(image_list) > 2
            
        except:
            return False
    
    def convert_table_to_natural_language(self, table: Dict) -> str:
        """Convert structured table data to natural language for better vector search"""
        if 'structured_data' not in table:
            return ""
        
        headers = table['structured_data']['headers']
        rows = table['structured_data']['rows']
        
        if not headers or not rows:
            return ""
        
        # Create natural language descriptions
        natural_text = []
        natural_text.append(f"Table {table['table_number']} from page {table['page']}:")
        
        for row in rows:
            if not any(cell for cell in row if cell and str(cell).strip()):
                continue  # Skip empty rows
                
            row_items = []
            for i, cell in enumerate(row):
                if i < len(headers) and cell and str(cell).strip():
                    header = str(headers[i]).strip()
                    value = str(cell).strip()
                    if header and value:
                        row_items.append(f"{header} is {value}")
            
            if row_items:
                natural_text.append(", ".join(row_items))
        
        return "\n".join(natural_text)
    
    def prepare_content_for_vectorization(self, page_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare content optimized for vector search while preserving display formats"""
        page_num = page_data["page_number"]
        
        # Content optimized for vector search (clean, natural language)
        vector_content_parts = []
        
        # Content with formatting preserved for display
        display_content_parts = []
        display_content_parts.append(f"=== PAGE {page_num} ===")
        
        # Metadata for tracking source types
        content_metadata = []
        
        # Process basic text
        if page_data["text_content"]:
            clean_text = page_data["text_content"].strip()
            vector_content_parts.append(clean_text)
            
            display_content_parts.append("--- TEXT CONTENT ---")
            display_content_parts.append(clean_text)
            
            content_metadata.append({
                "type": "text",
                "page": page_num,
                "source": "basic_extraction"
            })
        
        # Process tables - convert to natural language for vectors
        if page_data["tables"]:
            for table in page_data["tables"]:
                # Natural language for vector search
                natural_table_text = self.convert_table_to_natural_language(table)
                if natural_table_text:
                    vector_content_parts.append(natural_table_text)
                
                # Formatted version for display
                display_content_parts.append(f"--- TABLE {table['table_number']} (Page {table['page']}) ---")
                if "markdown_format" in table:
                    display_content_parts.append(table["markdown_format"])
                else:
                    display_content_parts.append(str(table.get("raw_data", "")))
                
                content_metadata.append({
                    "type": "table",
                    "page": page_num,
                    "table_number": table['table_number'],
                    "source": "table_extraction",
                    "formatted_version": table.get("markdown_format", ""),
                    "structured_data": table.get("structured_data", {})
                })
        
        # Process image text
        if page_data["images_text"]:
            for img_text in page_data["images_text"]:
                clean_image_text = img_text["extracted_text"].strip()
                if clean_image_text:
                    vector_content_parts.append(clean_image_text)
                
                display_content_parts.append(f"--- IMAGE TEXT {img_text['image_index']} (Page {img_text['page']}) ---")
                display_content_parts.append(clean_image_text)
                
                content_metadata.append({
                    "type": "image_text",
                    "page": page_num,
                    "image_index": img_text['image_index'],
                    "source": "image_ocr"
                })
        
        return {
            "vector_content": "\n\n".join(vector_content_parts),  # Clean for embeddings
            "display_content": "\n".join(display_content_parts),   # Formatted for display
            "content_metadata": content_metadata,
            "page_number": page_num
        }
    
    def combine_page_content(self, page_data: Dict[str, Any]) -> str:
        """Legacy method - now calls the optimized version"""
        optimized_content = self.prepare_content_for_vectorization(page_data)
        return optimized_content["display_content"]
    
    def process_pdf(self, pdf_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Main processing function with vector-optimized content preparation"""
        logger.info(f"Processing PDF with enhanced processor: {filename}")
        
        all_page_content = []
        optimized_pages = []  # For vector storage
        extraction_summary = {
            "pages_with_text": 0,
            "pages_with_tables": 0,
            "pages_with_images": 0,
            "total_tables": 0,
            "total_images": 0
        }
        
        try:
            # Determine number of pages
            doc = fitz.open("pdf", pdf_bytes)
            total_pages = min(doc.page_count, self.max_pages)
            doc.close()
            
            # Process each page
            for page_num in range(total_pages):
                logger.info(f"Processing page {page_num + 1}/{total_pages}")
                
                page_content = self.process_single_page(pdf_bytes, page_num)
                all_page_content.append(page_content)
                
                # Prepare optimized content for vectorization
                optimized_page = self.prepare_content_for_vectorization(page_content)
                optimized_pages.append(optimized_page)
                
                # Update summary
                if page_content["text_content"]:
                    extraction_summary["pages_with_text"] += 1
                if page_content["tables"]:
                    extraction_summary["pages_with_tables"] += 1
                    extraction_summary["total_tables"] += len(page_content["tables"])
                if page_content["images_text"]:
                    extraction_summary["pages_with_images"] += 1
                    extraction_summary["total_images"] += len(page_content["images_text"])
            
            # Combine content for vector storage (clean natural language)
            vector_content_parts = []
            display_content_parts = []
            
            for optimized_page in optimized_pages:
                if optimized_page["vector_content"].strip():
                    vector_content_parts.append(optimized_page["vector_content"])
                display_content_parts.append(optimized_page["display_content"])
            
            vector_optimized_content = "\n\n".join(vector_content_parts)
            display_formatted_content = "\n\n".join(display_content_parts)
            
            # Determine primary extraction method
            primary_method = "mixed"
            if extraction_summary["total_tables"] > 0:
                primary_method = "table_focused"
            elif extraction_summary["pages_with_images"] > extraction_summary["pages_with_text"]:
                primary_method = "image_focused"
            elif extraction_summary["pages_with_text"] > 0:
                primary_method = "text_focused"
            
            logger.info(f"Successfully processed {filename}: {len(vector_optimized_content)} chars for vectors, "
                       f"{extraction_summary['total_tables']} tables, {extraction_summary['total_images']} images")
            
            # 🔍 DEBUG: Print the entire extracted content to terminal
            print(f"\n{'='*80}")
            print(f"📄 EXTRACTED CONTENT FROM: {filename}")
            print(f"📊 Length: {len(vector_optimized_content)} characters")
            print(f"{'='*80}")
            print(vector_optimized_content)
            print(f"{'='*80}\n")
            
            return {
                "filename": filename,
                # Main content optimized for vector search (natural language)
                "total_content": vector_optimized_content,
                # Formatted content for display purposes
                "display_content": display_formatted_content,
                # Detailed page-by-page optimized data
                "optimized_pages": optimized_pages,
                "extraction_method": primary_method,
                "extraction_summary": extraction_summary,
                "page_details": all_page_content,  # Original detailed data
                "metadata": {
                    "page_count": total_pages,
                    "content_length": len(vector_optimized_content),
                    "display_length": len(display_formatted_content),
                    "has_tables": extraction_summary["total_tables"] > 0,
                    "has_images": extraction_summary["total_images"] > 0,
                    "extraction_methods_used": list(set(
                        method for page in all_page_content 
                        for method in page["extraction_methods"]
                    )),
                    "vector_optimized": True  # Flag indicating content is optimized for vectors
                }
            }
            
        except Exception as e:
            logger.error(f"PDF processing failed for {filename}: {e}")
            return {
                "filename": filename,
                "total_content": f"[Processing Error: {str(e)}]",
                "extraction_method": "error",
                "metadata": {"page_count": 0, "content_length": 0, "vector_optimized": False}
            }