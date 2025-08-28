import PyPDF2
from pdf2image import convert_from_bytes
import base64
from io import BytesIO
from openai import OpenAI
import os
from typing import List, Dict

class PDFProcessor:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def extract_text_basic(self, pdf_bytes: bytes) -> str:
        """Extract text using PyPDF2"""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text.strip():  # Only add non-empty pages
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            return text
        except Exception as e:
            print(f"⚠️ Basic text extraction failed: {e}")
            return ""
    
    def convert_to_images(self, pdf_bytes: bytes, max_pages: int = 10) -> List:
        """Convert PDF pages to images (limit pages for efficiency)"""
        try:
            images = convert_from_bytes(pdf_bytes, dpi=150, first_page=1, last_page=max_pages)
            print(f"📄 Converted {len(images)} pages to images")
            return images
        except Exception as e:
            print(f"⚠️ PDF to image conversion failed: {e}")
            return []
    
    def extract_with_vision(self, image, page_num: int) -> str:
        """Extract text, tables, and image content using OpenAI Vision"""
        try:
            # Convert PIL image to base64
            buffered = BytesIO()
            # Optimize image size
            image = image.resize((image.width // 2, image.height // 2), Image.LANCZOS)
            image.save(buffered, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            prompt = """
            Analyze this PDF page and extract ALL content including:
            
            1. **All visible text** - transcribe exactly as shown
            2. **Tables** - format as structured text with clear column/row separation
            3. **Headers and subheaders** - maintain hierarchy
            4. **Lists and bullet points** - preserve formatting
            5. **Any text in images, charts, or diagrams**
            6. **Captions and footnotes**
            
            Format your response as clean, structured text that preserves the original meaning and organization.
            Use clear paragraph breaks and maintain the logical flow of information.
            
            If there are tables, format them clearly with proper alignment.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # Updated to latest vision model
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                        ]
                    }
                ],
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"⚠️ Vision extraction failed for page {page_num}: {e}")
            return f"[Error processing page {page_num} with vision: {str(e)}]"
    
    def assess_text_quality(self, text: str) -> bool:
        """Assess if extracted text is of good quality"""
        if not text or len(text.strip()) < 50:
            return False
        
        # Check for reasonable text characteristics
        lines = text.strip().split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        if len(non_empty_lines) < 3:
            return False
        
        # Check if text has reasonable word density
        words = text.split()
        if len(words) < 20:
            return False
        
        # Check for too many special characters (might indicate extraction issues)
        special_char_ratio = sum(1 for char in text if not char.isalnum() and char not in ' \n\t.,!?:;()-') / len(text)
        if special_char_ratio > 0.3:
            return False
        
        return True
    
    def process_pdf(self, pdf_bytes: bytes, filename: str) -> Dict:
        """Main PDF processing function with intelligent fallback"""
        print(f"📋 Processing PDF: {filename}")
        
        # Try basic text extraction first
        basic_text = self.extract_text_basic(pdf_bytes)
        
        # Assess quality of basic extraction
        use_vision = not self.assess_text_quality(basic_text)
        
        if use_vision:
            print(f"📷 Basic extraction insufficient for {filename}, using vision...")
            
            # Convert to images and process with vision
            images = self.convert_to_images(pdf_bytes, max_pages=15)  # Process up to 15 pages
            
            vision_text_parts = []
            for i, image in enumerate(images):
                print(f"🔍 Processing page {i+1}/{len(images)} with vision...")
                page_content = self.extract_with_vision(image, i+1)
                if page_content and not page_content.startswith("[Error"):
                    vision_text_parts.append(f"\n=== Page {i+1} ===\n{page_content}")
            
            final_text = "\n".join(vision_text_parts) if vision_text_parts else basic_text
            extraction_method = "vision"
        else:
            final_text = basic_text
            extraction_method = "basic"
        
        # Ensure we have some content
        if not final_text or len(final_text.strip()) < 10:
            final_text = f"[Unable to extract readable content from {filename}. Please ensure the PDF contains text or clear images.]"
        
        result = {
            "filename": filename,
            "content": final_text,
            "extraction_method": extraction_method,
            "content_length": len(final_text),
            "page_count": len(final_text.split("=== Page")) if "=== Page" in final_text else basic_text.count("--- Page")
        }
        
        print(f"✅ Processed {filename}: {result['content_length']} characters, {result['page_count']} pages, method: {extraction_method}")
        
        return result

# Import PIL only if needed (for vision processing)
try:
    from PIL import Image
except ImportError:
    print("⚠️ PIL not available, vision processing may not work optimally")