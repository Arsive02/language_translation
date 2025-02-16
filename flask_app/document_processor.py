from typing import Dict, List, Optional, Union
import fitz  # PyMuPDF
import logging
from PIL import Image
import io
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, recognition_predictor, detection_predictor, langs=None):
        self.recognition_predictor = recognition_predictor
        self.detection_predictor = detection_predictor
        self.langs = langs or ["en"]
        self.supported_formats = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'}

    def _parse_page_range(self, page_range: str, max_pages: int) -> List[int]:
        """Parse page range string into list of page numbers"""
        if not page_range:
            return list(range(max_pages))
        
        pages = set()
        for part in page_range.split(','):
            if '-' in part:
                start, end = map(int, part.split('-'))
                pages.update(range(start, min(end + 1, max_pages)))
            else:
                page = int(part)
                if page < max_pages:
                    pages.add(page)
        return sorted(list(pages))

    def _process_page_with_ocr(self, image: Image.Image, page_number: int) -> Dict:
        """Process a page with OCR"""
        predictions = self.recognition_predictor(
            [image],
            [self.langs],
            self.detection_predictor
        )
        
        if not predictions or not predictions[0]:
            return {
                'text': '',
                'confidence': 0,
                'page_number': page_number
            }
            
        text_lines = []
        confidence_sum = 0
        
        for pred in predictions[0].text_lines:
            if pred.text.strip():
                text_lines.append({
                    'text': pred.text.strip(),
                    'confidence': pred.confidence,
                    'bbox': pred.bbox,
                    'polygon': pred.polygon
                })
            confidence_sum += pred.confidence
        
        avg_confidence = confidence_sum / len(text_lines) if text_lines else 0
        
        return {
            'text': ' '.join(line['text'] for line in text_lines),
            'text_lines': text_lines,
            'confidence': avg_confidence,
            'page_number': page_number,
            'size': image.size
        }

    def _process_pdf_with_ocr(self, pdf_doc, pages_to_process: List[int], save_images: bool, output_dir: Optional[str]) -> List[Dict]:
        """Process PDF pages using OCR"""
        results = []
        for page_num in pages_to_process:
            page = pdf_doc[page_num]
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            result = self._process_page_with_ocr(img, page_num)
            results.append(result)
            
            if save_images and output_dir:
                img_path = os.path.join(output_dir, f'page_{page_num}.png')
                pix.save(img_path)
        return results

    def _process_pdf_native(self, pdf_doc, pages_to_process: List[int]) -> List[Dict]:
        """Process PDF pages using native text extraction"""
        results = []
        for page_num in pages_to_process:
            page = pdf_doc[page_num]
            text = page.get_text()
            
            results.append({
                'text': text,
                'confidence': 1.0,  # Native extraction assumed to be confident
                'page_number': page_num,
                'text_lines': [{'text': text}],  # Simplified structure for native extraction
                'size': (page.rect.width, page.rect.height)
            })
        
        return results

    def process_document(
        self,
        file_data: bytes,
        filename: str,
        page_range: Optional[str] = None,
        save_images: bool = False,
        output_dir: Optional[str] = None,
        use_ocr: bool = False
    ) -> Dict:
        """
        Process document (PDF or image) and extract text
        
        Args:
            file_data: Raw file data
            filename: Original filename
            page_range: Page range for PDF processing (e.g., "0,5-10,20")
            save_images: Whether to save debug images
            output_dir: Directory to save debug images
            use_ocr: Whether to use OCR for text extraction
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            file_ext = Path(filename).suffix.lower()
            logger.info(f"Processing file: {filename} with extension: {file_ext}")
            
            if file_ext not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            if save_images and not output_dir:
                output_dir = os.path.join('debug_images', Path(filename).stem)
                os.makedirs(output_dir, exist_ok=True)
            
            # Process PDF
            if file_ext == '.pdf':
                with fitz.open(stream=file_data, filetype="pdf") as pdf_doc:
                    max_pages = len(pdf_doc)
                    pages_to_process = self._parse_page_range(page_range, max_pages)
                    
                    # Choose processing method based on use_ocr flag
                    if use_ocr:
                        results = self._process_pdf_with_ocr(pdf_doc, pages_to_process, save_images, output_dir)
                    else:
                        results = self._process_pdf_native(pdf_doc, pages_to_process)
                    
                    return {
                        'filename': filename,
                        'total_pages': max_pages,
                        'processed_pages': len(pages_to_process),
                        'pages': results,
                        'combined_text': ' '.join(page['text'] for page in results)
                    }
            
            # Process image (always use OCR)
            else:
                image = Image.open(io.BytesIO(file_data))
                result = self._process_page_with_ocr(image, 0)
                
                if save_images and output_dir:
                    img_path = os.path.join(output_dir, 'processed_image.png')
                    image.save(img_path)
                
                return {
                    'filename': filename,
                    'total_pages': 1,
                    'processed_pages': 1,
                    'pages': [result] if result['text'] else [],
                    'combined_text': result['text']
                }
                
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise