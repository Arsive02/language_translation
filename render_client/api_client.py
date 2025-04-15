import logging
import os
import time
from typing import Any, Dict

import requests
from dotenv import load_dotenv

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
        self.api_url = api_url or os.getenv("HF_API_URL", "https://arsive-lt-space.hf.space")
        logger.info(f"Initialized Translation Client with API URL: {self.api_url}")
    
    def translate_text(self, text: str, source_lang_code: str, target_lang_code: str, timeout: int = 60) -> Dict[str, Any]:
        """
        Translate text using the API
        
        Args:
            text: Text to translate
            source_lang_code: Source language code
            target_lang_code: Target language code
            timeout: Request timeout in seconds
            
        Returns:
            Dictionary with translation results
        """
        try:
            start_time = time.time()
            logger.info(f"Sending translation request: {source_lang_code} → {target_lang_code}, length: {len(text)} chars")
            
            endpoint = f"{self.api_url}/translate"
            payload = {
                "text": text,
                "source_lang_code": source_lang_code,
                "target_lang_code": target_lang_code
            }
            
            response = requests.post(endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            
            duration = time.time() - start_time
            logger.info(f"Translation completed in {duration:.2f}s: {source_lang_code} → {target_lang_code}")
            
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Translation API error: {str(e)}")
            raise
    
    def translate_html(self, html: str, source_lang_code: str, target_lang_code: str, timeout: int = 120) -> Dict[str, Any]:
        """
        Translate HTML using the API
        
        Args:
            html: HTML content to translate
            source_lang_code: Source language code
            target_lang_code: Target language code
            timeout: Request timeout in seconds (longer for HTML)
            
        Returns:
            Dictionary with translation results
        """
        try:
            start_time = time.time()
            logger.info(f"Sending HTML translation request: {source_lang_code} → {target_lang_code}, length: {len(html)} chars")
            
            endpoint = f"{self.api_url}/translate-html"
            payload = {
                "html": html,
                "source_lang_code": source_lang_code,
                "target_lang_code": target_lang_code
            }
            
            response = requests.post(endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            
            duration = time.time() - start_time
            logger.info(f"HTML translation completed in {duration:.2f}s: {source_lang_code} → {target_lang_code}")
            
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
        use_ocr: bool = False,
        timeout: int = 180
    ) -> Dict[str, Any]:
        """
        Process and translate a document using the API
        
        Args:
            file_data: Raw file content
            filename: Original filename
            source_lang_code: Source language code
            target_lang_code: Target language code
            use_ocr: Whether to use OCR
            timeout: Request timeout in seconds (longest for documents)
            
        Returns:
            Dictionary with document processing results
        """
        try:
            start_time = time.time()
            logger.info(f"Sending document processing request: {filename}, {source_lang_code} → {target_lang_code}")
            
            endpoint = f"{self.api_url}/process-document"
            
            files = {
                'file': (filename, file_data)
            }
            
            data = {
                'source_lang_code': source_lang_code,
                'target_lang_code': target_lang_code,
                'use_ocr': str(use_ocr).lower()
            }
            
            response = requests.post(endpoint, files=files, data=data, timeout=timeout)
            response.raise_for_status()
            
            duration = time.time() - start_time
            logger.info(f"Document processing completed in {duration:.2f}s: {filename}")
            
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Document processing API error: {str(e)}")
            raise
            
    def health_check(self, timeout: int = 10) -> Dict[str, Any]:
        """
        Check the health of the translation API
        
        Args:
            timeout: Request timeout in seconds
            
        Returns:
            Health check results
        """
        try:
            endpoint = f"{self.api_url}/health"
            response = requests.get(endpoint, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Health check error: {str(e)}")
            return {"status": "error", "message": str(e)}
        