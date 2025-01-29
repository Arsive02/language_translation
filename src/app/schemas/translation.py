from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class LanguageFamily(str, Enum):
    DRAVIDIAN = "Dravidian"
    SLAVIC = "Slavic"
    GERMANIC = "Germanic"
    ROMANCE = "Romance"

class TextTranslationRequest(BaseModel):
    text: str = Field(..., description="Text to translate")
    source_lang: str = Field(..., description="Source language")
    target_lang: str = Field(..., description="Target language")
    is_family: bool = Field(False, description="Whether target language is from a language family")
    family_name: Optional[str] = Field(None, description="Language family name if is_family is True")

class TranslationResponse(BaseModel):
    translated_text: str
    source_lang: str
    target_lang: str
    error: Optional[str] = None

class ImageTranslationResponse(BaseModel):
    extracted_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    error: Optional[str] = None

class LanguageInfo(BaseModel):
    families: dict
    individual_languages: dict

class Point(BaseModel):
    x: float
    y: float

class Polygon(BaseModel):
    points: List[Point]

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class TextLine(BaseModel):
    text: str
    confidence: float
    polygon: List[tuple]
    bbox: tuple

class PageResult(BaseModel):
    text_lines: List[TextLine]
    languages: List[str]
    page: int
    image_bbox: tuple

class OCRRequest(BaseModel):
    languages: Optional[List[str]] = Field(None, description="List of languages for OCR")
    page_range: Optional[str] = Field(None, description="Page range for PDF processing")
    save_images: bool = Field(False, description="Whether to save debug images")