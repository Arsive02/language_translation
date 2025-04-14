import requests
import logging
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class TranslationClient:
    """Client for the Hugging Face Spaces API"""
    
    def __init__(self, api_url=None):
        """
        Initialize the API client
        
        Args:
            api_url: URL of the Hugging Face Spaces API (or from env var)
        """
        self.api_url = api_url or os.getenv("HF_API_URL", "https://username-universal-translator.hf.space")
        logger.info(f"Initialized Translation Client with API URL: {self.api_url}")
    
    def translate_text(self, text: str, source_lang_code: str, target_lang_code: str) -> Dict[str, Any]:
        """
        Translate text using the API
        
        Args:
            text: Text to translate
            source_lang_code: Source language code
            target_lang_code: Target language code
            
        Returns:
            Dictionary with translation results
        """
        try:
            endpoint = f"{self.api_url}/translate"
            payload = {
                "text": text,
                "source_lang_code": source_lang_code,
                "target_lang_code": target_lang_code
            }
            
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()
            
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Translation API error: {str(e)}")
            raise
    
    def translate_html(self, html: str, source_lang_code: str, target_lang_code: str) -> Dict[str, Any]:
        """
        Translate HTML using the API
        
        Args:
            html: HTML content to translate
            source_lang_code: Source language code
            target_lang_code: Target language code
            
        Returns:
            Dictionary with translation results
        """
        try:
            endpoint = f"{self.api_url}/translate-html"
            payload = {
                "html": html,
                "source_lang_code": source_lang_code,
                "target_lang_code": target_lang_code
            }
            
            response = requests.post(endpoint, json=payload)
            response.raise_for_status()
            
            return response.json()
        except requests.RequestException as e:
            logger.error(f"HTML translation API error: {str(e)}")
            raise
    
    def process_document(
        self, 
        file_data: bytes, 
        filename: str, 
        source_lang_code: str, 
        target_lang_code: str,
        use_ocr: bool = False
    ) -> Dict[str, Any]:
        """
        Process and translate a document using the API
        
        Args:
            file_data: Raw file content
            filename: Original filename
            source_lang_code: Source language code
            target_lang_code: Target language code
            use_ocr: Whether to use OCR
            
        Returns:
            Dictionary with document processing results
        """
        try:
            endpoint = f"{self.api_url}/process-document"
            
            files = {
                'file': (filename, file_data)
            }
            
            data = {
                'source_lang_code': source_lang_code,
                'target_lang_code': target_lang_code,
                'use_ocr': str(use_ocr).lower()
            }
            
            response = requests.post(endpoint, files=files, data=data)
            response.raise_for_status()
            
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Document processing API error: {str(e)}")
            raise