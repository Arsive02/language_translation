from typing import Dict, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File
from app.schemas.translation import (
    TextTranslationRequest, 
    TranslationResponse, 
    ImageTranslationResponse,
    LanguageInfo
)
from app.services.translator import TranslationService
from app.services.ocr import OCRService
from app.core.constants import LANGUAGE_FAMILIES, INDIVIDUAL_LANGUAGES
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

translation_service = TranslationService()
ocr_service = OCRService()

@router.get("/languages", response_model=LanguageInfo)
async def get_available_languages():
    """Get all available languages and language families"""
    return {
        "families": LANGUAGE_FAMILIES,
        "individual_languages": INDIVIDUAL_LANGUAGES
    }

@router.post("/translate/text", response_model=TranslationResponse)
async def translate_text(request: TextTranslationRequest):
    """Translate text between languages"""
    try:
        # Get language info
        source_lang_info = INDIVIDUAL_LANGUAGES[request.source_lang]
        if request.is_family:
            target_lang_info = LANGUAGE_FAMILIES[request.family_name][request.target_lang]
        else:
            target_lang_info = INDIVIDUAL_LANGUAGES[request.target_lang]
        
        translated_text = await translation_service.translate(
            request.text,
            source_lang_info,
            target_lang_info
        )
        
        return TranslationResponse(
            translated_text=translated_text,
            source_lang=request.source_lang,
            target_lang=request.target_lang
        )
        
    except Exception as e:
        logger.error(f"Translation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Translation failed: {str(e)}"
        )

@router.post("/ocr", response_model=Dict)
async def perform_ocr(
    file: UploadFile = File(...),
    languages: Optional[str] = None,
    page_range: Optional[str] = None,
    save_images: bool = False
):
    """
    Perform OCR on uploaded document
    - languages: comma-separated list of language codes
    - page_range: page range for PDF processing (e.g., "0,5-10,20")
    - save_images: whether to save debug images
    """
    try:
        # Parse languages
        lang_list = None
        if languages:
            lang_list = [lang.strip() for lang in languages.split(',')]
        
        # Read file
        file_data = await file.read()
        
        # Process document
        results = await ocr_service.process_document(
            file_data,
            file.filename,
            languages=lang_list,
            page_range=page_range,
            save_images=save_images
        )
        
        return results
        
    except Exception as e:
        logger.error(f"OCR failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"OCR failed: {str(e)}"
        )