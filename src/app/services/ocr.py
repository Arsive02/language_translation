import logging
from PIL import Image
import fitz  # PyMuPDF for PDF processing
from io import BytesIO
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from pathlib import Path
import tempfile
import os
import json
from typing import List, Dict, Optional, Union, BinaryIO

logger = logging.getLogger(__name__)

class OCRService:
    def __init__(self):
        self._initialize_predictors()
        self.supported_formats = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'}

    def _initialize_predictors(self):
        try:
            self.recognition_predictor = RecognitionPredictor()
            self.detection_predictor = DetectionPredictor()
            logger.info("OCR predictors initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing OCR predictors: {str(e)}")
            raise

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

    def _process_pdf(
        self,
        pdf_data: bytes,
        languages: List[str],
        page_range: Optional[str] = None,
        save_images: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict:
        """Process PDF document and extract text"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(pdf_data)
            tmp_file.flush()
            
            try:
                pdf_document = fitz.open(tmp_file.name)
                max_pages = len(pdf_document)
                pages_to_process = self._parse_page_range(page_range, max_pages)
                
                results = []
                for page_num in pages_to_process:
                    page = pdf_document[page_num]
                    pix = page.get_pixmap()
                    
                    # Convert to PIL Image
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # Run OCR
                    predictions = self.recognition_predictor([img], [languages], self.detection_predictor)
                    
                    if predictions and len(predictions) > 0:
                        page_result = {
                            'text_lines': [],
                            'languages': languages,
                            'page': page_num,
                            'image_bbox': (0, 0, pix.width, pix.height)
                        }
                        
                        for pred in predictions[0]:
                            text_line = {
                                'text': pred.text,
                                'confidence': pred.confidence,
                                'polygon': pred.polygon,
                                'bbox': pred.bbox
                            }
                            page_result['text_lines'].append(text_line)
                        
                        results.append(page_result)
                        
                        if save_images and output_dir:
                            img_path = os.path.join(output_dir, f'page_{page_num}.png')
                            img.save(img_path)
                
                return results
                
            finally:
                os.unlink(tmp_file.name)

    def _process_image(
        self,
        image_data: bytes,
        languages: List[str],
        save_images: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict:
        """Process single image and extract text"""
        image = Image.open(BytesIO(image_data))
        predictions = self.recognition_predictor([image], [languages], self.detection_predictor)
        
        if predictions and len(predictions) > 0:
            width, height = image.size
            result = {
                'text_lines': [],
                'languages': languages,
                'page': 0,
                'image_bbox': (0, 0, width, height)
            }
            
            for pred in predictions[0]:
                text_line = {
                    'text': pred.text,
                    'confidence': pred.confidence,
                    'polygon': pred.polygon,
                    'bbox': pred.bbox
                }
                result['text_lines'].append(text_line)
            
            if save_images and output_dir:
                img_path = os.path.join(output_dir, 'processed_image.png')
                image.save(img_path)
            
            return [result]
        
        return []

    async def process_document(
        self,
        file_data: bytes,
        filename: str,
        languages: Optional[List[str]] = None,
        page_range: Optional[str] = None,
        save_images: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict:
        """Process document (PDF or image) and extract text"""
        try:
            if not languages:
                languages = ["en"]
                
            if not output_dir and save_images:
                output_dir = tempfile.mkdtemp()
                
            file_ext = Path(filename).suffix.lower()
            if file_ext not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_ext}")
                
            if file_ext == '.pdf':
                results = self._process_pdf(
                    file_data,
                    languages,
                    page_range,
                    save_images,
                    output_dir
                )
            else:
                results = self._process_image(
                    file_data,
                    languages,
                    save_images,
                    output_dir
                )
                
            return {
                Path(filename).stem: results
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise