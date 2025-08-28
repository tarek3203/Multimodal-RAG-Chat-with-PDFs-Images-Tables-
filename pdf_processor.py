import PyPDF2
from pdf2image import convert_from_bytes
import base64
from io import BytesIO
from langchain_openai import OpenAI
from langchain.schema import HumanMessage
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
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Basic text extraction failed: {e}")
            return ""
    
    def convert_to_images(self, pdf_bytes: bytes) -> List:
        """Convert PDF pages to images"""
        try:
            images = convert_from_bytes(pdf_bytes, dpi=200)
            return images
        except Exception as e:
            print(f"PDF to image conversion failed: {e}")
            return []
    
    def extract_with_vision(self, image) -> str:
        """Extract text, tables, and image content using OpenAI Vision"""
        try:
            # Convert PIL image to base64
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            prompt = """
            Analyze this PDF page image and extract ALL content including:
            1. All visible text (including text in images/charts)
            2. Table data in structured format
            3. Any embedded text from images or diagrams
            4. Document structure (headers, paragraphs, lists)
            
            Format the response as clear, structured text that preserves the original meaning and organization.
            """
            
            # Note: Using OpenAI's vision capabilities
            # You might need to use openai client directly for vision
            response = self.openai_client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                        ]
                    }
                ],
                max_tokens=1500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Vision extraction failed: {e}")
            return ""
    
    def process_pdf(self, pdf_bytes: bytes, filename: str) -> Dict:
        """Main PDF processing function"""
        print(f"Processing PDF: {filename}")
        
        # Try basic text extraction first
        basic_text = self.extract_text_basic(pdf_bytes)
        
        # If basic extraction is insufficient, use vision
        vision_text = ""
        if len(basic_text.strip()) < 100:  # If basic extraction is poor
            print("Basic extraction insufficient, using vision...")
            images = self.convert_to_images(pdf_bytes)
            for i, image in enumerate(images[:5]):  # Process first 5 pages
                page_content = self.extract_with_vision(image)
                vision_text += f"\n--- Page {i+1} ---\n{page_content}\n"
        
        # Combine all extracted content
        final_text = basic_text if basic_text.strip() else vision_text
        
        return {
            "filename": filename,
            "content": final_text,
            "page_count": len(basic_text.split('\n')) if basic_text else len(vision_text.split('--- Page'))
        }