# ===== services/ocr_processor.py =====
import torch
from PIL import Image
import numpy as np
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)

class OCRProcessor:
    """Unified OCR processor supporting multiple models optimized for Mac M1"""
    
    def __init__(self, method: str = "trocr"):
        self.method = method.lower()
        self.model = None
        self.processor = None
        self.reader = None
        
        # Force CPU for M1 compatibility
        self.device = "cpu"
        
        self._load_model()
    
    def _load_model(self):
        """Load the selected OCR model"""
        try:
            if self.method == "trocr":
                self._load_trocr()
            elif self.method == "easyocr":
                self._load_easyocr()
            elif self.method == "got_ocr":
                self._load_got_ocr()
            else:
                raise ValueError(f"Unsupported OCR method: {self.method}")
                
            logger.info(f"Successfully loaded {self.method} OCR model")
            
        except Exception as e:
            logger.error(f"Failed to load {self.method} model: {e}")
            # Fallback to EasyOCR
            if self.method != "easyocr":
                logger.info("Falling back to EasyOCR...")
                self.method = "easyocr"
                self._load_easyocr()
    
    def _load_trocr(self):
        """Load TrOCR model (lightweight, ~300MB)"""
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        
        model_name = "microsoft/trocr-base-printed"
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
    
    def _load_easyocr(self):
        """Load EasyOCR (good fallback option)"""
        import easyocr
        self.reader = easyocr.Reader(['en'], gpu=False)  # Force CPU
    
    def _load_got_ocr(self):
        """Load GOT-OCR2.0 (best quality but larger)"""
        from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
        
        model_name = "stepfun-ai/GOT-OCR2_0"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map=None  # Force CPU
        )
        self.model.to(self.device)
        self.model.eval()
    
    def extract_text(self, image: Union[Image.Image, np.ndarray, str]) -> str:
        """
        Extract text from an image using the loaded OCR model
        
        Args:
            image: PIL Image, numpy array, or file path
            
        Returns:
            Extracted text string
        """
        # Convert input to PIL Image if needed
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        try:
            if self.method == "trocr":
                return self._extract_with_trocr(image)
            elif self.method == "easyocr":
                return self._extract_with_easyocr(image)
            elif self.method == "got_ocr":
                return self._extract_with_got_ocr(image)
            else:
                return ""
                
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return f"[OCR Error: {str(e)}]"
    
    def _extract_with_trocr(self, image: Image.Image) -> str:
        """Extract text using TrOCR"""
        with torch.no_grad():
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            generated_ids = self.model.generate(pixel_values, max_length=512)
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
        return text.strip()
    
    def _extract_with_easyocr(self, image: Image.Image) -> str:
        """Extract text using EasyOCR"""
        # Convert PIL to numpy for EasyOCR
        img_array = np.array(image)
        results = self.reader.readtext(img_array)
        
        # Combine all detected text
        text_parts = [result[1] for result in results if result[1].strip()]
        return " ".join(text_parts)
    
    def _extract_with_got_ocr(self, image: Image.Image) -> str:
        """Extract text using GOT-OCR2.0"""
        with torch.no_grad():
            # Prepare prompt for OCR
            prompt = "OCR with format"  # GOT-OCR2.0 prompt
            
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            generated_ids = self.model.generate(**inputs, max_new_tokens=1000)
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
        return text.strip()
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            "method": self.method,
            "device": self.device,
            "available": self.model is not None or self.reader is not None
        }