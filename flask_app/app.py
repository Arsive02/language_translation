# Standard library imports
import logging
import os
import re
import torch
from flask import Flask, render_template, request, jsonify
from transformers import T5ForConditionalGeneration, T5Tokenizer
from bs4 import BeautifulSoup
from tqdm import tqdm

from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from document_processor import DocumentProcessor
from text_chunker import TextChunker
from lt_logger import TranslationLogger
from html_processor import HTMLProcessor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

translation_logger = TranslationLogger(log_dir="translation_logs")

# Language configurations for MADLAD-400 model
LANGUAGE_CODES = {
    # MADLAD-400 language codes (using 2-letter ISO codes)
    'English': 'en',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Italian': 'it',
    'Portuguese': 'pt',
    'Dutch': 'nl',
    'Polish': 'pl',
    'Russian': 'ru',
    'Chinese': 'zh',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Arabic': 'ar',
    'Hindi': 'hi',
    'Tamil': 'ta',
    'Telugu': 'te',
    'Kannada': 'kn',
    'Malayalam': 'ml',
    'Czech': 'cs',
    'Slovak': 'sk',
    'Swedish': 'sv',
    'Danish': 'da'
}

LANGUAGE_FAMILIES = {
    'Dravidian': {
        'Tamil': 'ta',
        'Telugu': 'te',
        'Kannada': 'kn',
        'Malayalam': 'ml'
    },
    'Slavic': {
        'Russian': 'ru',
        'Polish': 'pl',
        'Czech': 'cs',
        'Slovak': 'sk'
    },
    'Germanic': {
        'German': 'de',
        'Dutch': 'nl',
        'Swedish': 'sv',
        'Danish': 'da'
    },
    'Romance': {
        'Spanish': 'es',
        'French': 'fr',
        'Italian': 'it',
        'Portuguese': 'pt'
    },
    'Asian': {
        'Chinese': 'zh',
        'Japanese': 'ja',
        'Korean': 'ko',
    },
    'Indic': {
        'Hindi': 'hi',
        'Bengali': 'bn',
        'Marathi': 'mr',
        'Gujarati': 'gu'
    }
}

INDIVIDUAL_LANGUAGES = {lang: code for lang, code in LANGUAGE_CODES.items()}

class TranslationService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        self.recognition_predictor = RecognitionPredictor()
        self.detection_predictor = DetectionPredictor()
        self.langs = ["en"]
        self.document_processor = DocumentProcessor(
            self.recognition_predictor,
            self.detection_predictor,
            self.langs
        )
        self.text_chunker = TextChunker(max_tokens=250, overlap_tokens=30)
        self.html_processor = HTMLProcessor()
        
        # Load model during initialization to avoid reloading
        self._load_model()
        
    def _get_device(self):
        """Get the best available device for model inference."""
        if torch.cuda.is_available():
            logger.info("Using CUDA GPU for translation")
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Using Apple MPS (Metal) for translation")
            return torch.device("mps")
        else:
            logger.info("Using CPU for translation")
            return torch.device("cpu")
            
    def _load_model(self):
        """Load the MADLAD-400 3B translation model."""
        try:
            model_name = "google/madlad400-3b-mt"
            
            logger.info(f"Loading translation model: {model_name}")
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            
            # Use torch_dtype=torch.bfloat16 if available for faster inference
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                logger.info("Using bfloat16 precision for model loading")
                self.model = T5ForConditionalGeneration.from_pretrained(
                    model_name, 
                    torch_dtype=torch.bfloat16
                )
            else:
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                logger.info(f"Using {dtype} precision for model loading")
                self.model = T5ForConditionalGeneration.from_pretrained(
                    model_name, 
                    torch_dtype=dtype
                )
            
            self.model.to(self.device)
            
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            
    def translate_text(self, text, source_lang_code, target_lang_code):
        """Translate text using chunking for long texts with MADLAD-400 model."""
        try:
            if self.model is None or self.tokenizer is None:
                return "Error: Translation model not loaded"
            
            # Get chunks using TextChunker
            chunks = self.text_chunker.create_chunks(text)
            translated_chunks = []

            # Translate each chunk
            for chunk in tqdm(chunks, desc="Translating chunks", unit="chunk"):
                try:
                    # Prepare input with MADLAD-400 format: <2{target_lang}> {source_text}
                    # This prepends the target language token to the source text
                    input_text = f"<2{target_lang_code}> {chunk.text}"
                    
                    inputs = self.tokenizer(
                        input_text, 
                        return_tensors="pt", 
                        padding=True,
                        truncation=True,
                        max_length=512
                    )
                    
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        translated = self.model.generate(
                            **inputs,
                            max_length=512,
                            num_beams=5,
                            early_stopping=True
                        )
                    
                    translated_text = self.tokenizer.batch_decode(
                        translated, 
                        skip_special_tokens=True
                    )[0]
                    
                    translated_chunks.append(translated_text)
                except Exception as e:
                    logger.error(f"Error translating part {chunk.index + 1}: {str(e)}")
                    translated_chunks.append(f"[Error translating part {chunk.index + 1}]")

            # Combine translations using TextChunker
            final_translation = self.text_chunker.combine_translations(text, chunks, translated_chunks)
            return re.sub(r'\s+', ' ', final_translation).strip()

        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return f"Translation error: {str(e)}"
            
    def translate_html(self, html_content, source_lang_code, target_lang_code):
        """Translate HTML content while preserving HTML structure using MADLAD-400 model."""
        try:
            # Extract text and HTML structure
            text_fragments, html_map = self.html_processor.extract_text(html_content)
            
            if not text_fragments:
                return html_content  # No text to translate
            
            # Process each text fragment individually for better translation quality
            translated_fragments = []
            for fragment in tqdm(text_fragments, desc="Translating HTML fragments", unit="fragment"):
                if not fragment.strip():
                    translated_fragments.append(fragment)
                    continue
                
                # Translate each fragment separately
                translated_fragment = self.translate_text(fragment, source_lang_code, target_lang_code)
                translated_fragments.append(translated_fragment)
            
            # Replace the original text with translated text in the HTML structure
            translated_html = self.html_processor.replace_text(html_map, translated_fragments)
            
            return translated_html
            
        except Exception as e:
            logger.error(f"HTML translation error: {str(e)}")
            return f"HTML translation error: {str(e)}"

    def process_document(self, file_data: bytes, filename: str, use_ocr: bool = False):
        """Process document using DocumentProcessor."""
        try:
            result = self.document_processor.process_document(
                file_data=file_data,
                filename=filename,
                save_images=False, 
                use_ocr=use_ocr
            )
            return result.get('combined_text', '')
        except Exception as e:
            logger.error(f"Document processing error: {str(e)}")
            return None

translation_service = TranslationService()

@app.route('/')
def index():
    return render_template(
        'index.html',
        language_families=LANGUAGE_FAMILIES,
        individual_languages=INDIVIDUAL_LANGUAGES
    )

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.form
        text = data.get('text', '')
        source_lang = data.get('source_lang', 'English')
        target_lang = data.get('target_lang', '')
        is_family = str(data.get('is_family')).lower() == 'true'
        family_name = data.get('family_name', '')

        # Validate source language
        if source_lang not in INDIVIDUAL_LANGUAGES:
            raise ValueError(f"Invalid source language: {source_lang}")
        
        source_lang_code = INDIVIDUAL_LANGUAGES[source_lang]
        
        # Handle language family translation
        if is_family:
            if family_name not in LANGUAGE_FAMILIES:
                raise ValueError(f"Invalid language family: {family_name}")
            if target_lang not in LANGUAGE_FAMILIES[family_name]:
                raise ValueError(f"Invalid target language {target_lang} for family {family_name}")
            target_lang_code = LANGUAGE_FAMILIES[family_name][target_lang]
        else:
            if target_lang not in INDIVIDUAL_LANGUAGES:
                raise ValueError(f"Invalid target language: {target_lang}")
            if target_lang == source_lang:
                raise ValueError("Source and target languages cannot be the same")
            target_lang_code = INDIVIDUAL_LANGUAGES[target_lang]

        translated_text = translation_service.translate_text(
            text,
            source_lang_code,
            target_lang_code
        )

        translation_logger.log_translation(
            source_text=text,
            translated_text=translated_text,
            source_lang=source_lang,
            target_lang=target_lang,
            is_family=is_family,
            family_name=family_name,
            success=True
        )

        return jsonify({
            'success': True,
            'translated_text': translated_text
        })

    except (ValueError, Exception) as e:
        error_message = str(e)
        logger.error(f"Translation error: {error_message}")
        
        translation_logger.log_translation(
            source_text=data.get('text', '') if 'data' in locals() else '',
            translated_text="",
            source_lang=data.get('source_lang', '') if 'data' in locals() else '',
            target_lang=data.get('target_lang', '') if 'data' in locals() else '',
            is_family=str(data.get('is_family', '')).lower() == 'true' if 'data' in locals() else False,
            family_name=data.get('family_name', '') if 'data' in locals() else '',
            success=False,
            error_message=error_message
        )
        
        return jsonify({
            'success': False,
            'error': error_message
        })

@app.route('/translate-html', methods=['POST'])
def translate_html():
    try:
        data = request.form
        html_content = data.get('html', '')
        source_lang = data.get('source_lang', 'English')
        target_lang = data.get('target_lang', '')
        is_family = str(data.get('is_family')).lower() == 'true'
        family_name = data.get('family_name', '')

        # Validate source language
        if source_lang not in INDIVIDUAL_LANGUAGES:
            raise ValueError(f"Invalid source language: {source_lang}")
        
        source_lang_code = INDIVIDUAL_LANGUAGES[source_lang]
        
        # Handle language family translation
        if is_family:
            if family_name not in LANGUAGE_FAMILIES:
                raise ValueError(f"Invalid language family: {family_name}")
            if target_lang not in LANGUAGE_FAMILIES[family_name]:
                raise ValueError(f"Invalid target language {target_lang} for family {family_name}")
            target_lang_code = LANGUAGE_FAMILIES[family_name][target_lang]
        else:
            if target_lang not in INDIVIDUAL_LANGUAGES:
                raise ValueError(f"Invalid target language: {target_lang}")
            if target_lang == source_lang:
                raise ValueError("Source and target languages cannot be the same")
            target_lang_code = INDIVIDUAL_LANGUAGES[target_lang]

        translated_html = translation_service.translate_html(
            html_content,
            source_lang_code,
            target_lang_code
        )

        translation_logger.log_translation(
            source_text=html_content,
            translated_text=translated_html,
            source_lang=source_lang,
            target_lang=target_lang,
            is_family=is_family,
            family_name=family_name,
            success=True,
            translation_type="html"
        )

        return jsonify({
            'success': True,
            'translated_html': translated_html
        })

    except (ValueError, Exception) as e:
        error_message = str(e)
        logger.error(f"HTML translation error: {error_message}")
        
        translation_logger.log_translation(
            source_text=data.get('html', '') if 'data' in locals() else '',
            translated_text="",
            source_lang=data.get('source_lang', '') if 'data' in locals() else '',
            target_lang=data.get('target_lang', '') if 'data' in locals() else '',
            is_family=str(data.get('is_family', '')).lower() == 'true' if 'data' in locals() else False,
            family_name=data.get('family_name', '') if 'data' in locals() else '',
            success=False,
            error_message=error_message,
            translation_type="html"
        )
        
        return jsonify({
            'success': False,
            'error': error_message
        })

@app.route('/process-document', methods=['POST'])
def process_document():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})

        use_ocr = request.form.get('use_ocr', 'false').lower() == 'true'
        source_lang = request.form.get('source_lang', 'English')
        target_lang = request.form.get('target_lang', '')
        is_family = request.form.get('is_family') == 'true'
        family_name = request.form.get('family_name', '')

        # Process document
        file_data = file.read()
        extracted_text = translation_service.process_document(
            file_data=file_data,
            filename=file.filename,
            use_ocr=use_ocr
        )

        if extracted_text:
            # Handle translation
            source_lang_code = INDIVIDUAL_LANGUAGES[source_lang]
            if is_family:
                target_lang_code = LANGUAGE_FAMILIES[family_name][target_lang]
            else:
                target_lang_code = INDIVIDUAL_LANGUAGES[target_lang]

            translated_text = translation_service.translate_text(
                extracted_text,
                source_lang_code,
                target_lang_code
            )

            translation_logger.log_translation(
                source_text=extracted_text,
                translated_text=translated_text,
                source_lang=source_lang,
                target_lang=target_lang,
                is_family=is_family,
                family_name=family_name,
                extracted_text=extracted_text,
                file_name=file.filename,
                success=True
            )

            return jsonify({
                'success': True,
                'result': {
                    'extracted_text': extracted_text,
                    'translated_text': translated_text
                }
            })
        else:
            translation_logger.log_translation(
                source_text="",
                translated_text="",
                source_lang=source_lang,
                target_lang=target_lang,
                is_family=is_family,
                family_name=family_name,
                extracted_text="",
                file_name=file.filename,
                success=False,
                error_message="No text extracted"
            )
            return jsonify({
                'success': False,
                'error': "No text could be extracted from the document"
            })

    except Exception as e:
        error_message = str(e)
        logger.error(f"Document processing error: {error_message}")
        
        translation_logger.log_translation(
            source_text="",
            translated_text="",
            source_lang=request.form.get('source_lang', 'English'),
            target_lang=request.form.get('target_lang', ''),
            is_family=request.form.get('is_family') == 'true',
            family_name=request.form.get('family_name', ''),
            file_name=file.filename if 'file' in locals() else None,
            success=False,
            error_message=error_message
        )
        
        return jsonify({
            'success': False,
            'error': error_message
        })
    
@app.route('/logs', methods=['GET'])
def get_logs():
    date = request.args.get('date')
    logs = translation_logger.get_logs(date)
    return jsonify(logs)

@app.route('/logs/search', methods=['GET'])
def search_logs():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    translation_type = request.args.get('type')
    source_lang = request.args.get('source_lang')
    target_lang = request.args.get('target_lang')
    success_only = request.args.get('success_only', 'false').lower() == 'true'

    logs = translation_logger.search_logs(
        start_date=start_date,
        end_date=end_date,
        translation_type=translation_type,
        source_lang=source_lang,
        target_lang=target_lang,
        success_only=success_only
    )
    return jsonify(logs)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)