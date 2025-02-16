# Standard library imports
import logging
import os
import re
from flask import Flask, render_template, request, jsonify
from transformers import MarianMTModel, MarianTokenizer

from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from document_processor import DocumentProcessor
from text_chunker import TextChunker

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Language configurations
LANGUAGE_FAMILIES = {
    'Dravidian': {
        'Tamil': ('en', 'dra', '>>tam<<'),
        'Telugu': ('en', 'dra', '>>tel<<'),
        'Kannada': ('en', 'dra', '>>kan<<'),
        'Malayalam': ('en', 'dra', '>>mal<<')
    },
    'Slavic': {
        'Russian': ('en', 'ru', '>>rus<<'),
        'Polish': ('eng', 'pol', '>>pol<<'),
        'Czech': ('en', 'cs', '>>ces<<'),
        'Slovak': ('en', 'sk', '>>sk<<')
    },
    'Germanic': {
        'German': ('en', 'deu', '>>deu<<'),
        'Dutch': ('en', 'nld', '>>nld<<'),
        'Swedish': ('en', 'swe', '>>swe<<'),
        'Danish': ('en', 'dan', '>>dan<<')
    },
    'Romance': {
        'Spanish': ('eng', 'spa', '>>spa<<'),
        'French': ('en', 'fr', '>>fra<<'),
        'Italian': ('eng', 'ita', '>>ita<<'),
        'Portuguese': ('eng', 'por', '>>por<<')
    }
}

INDIVIDUAL_LANGUAGES = {
    'English': ('en', 'en', '>>eng<<'),
    'Spanish': ('en', 'es', '>>spa<<'),
    'French': ('en', 'fr', '>>fra<<'),
    'German': ('en', 'deu', '>>deu<<'),
    'Italian': ('en', 'ita', '>>ita<<'),
    'Portuguese': ('en', 'por', '>>por<<'),
    'Dutch': ('en', 'nld', '>>nld<<'),
    'Polish': ('en', 'pol', '>>pol<<'),
    'Russian': ('en', 'rus', '>>rus<<')
}

class TranslationService:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.recognition_predictor = RecognitionPredictor()
        self.detection_predictor = DetectionPredictor()
        self.langs = ["en"]
        self.document_processor = DocumentProcessor(
            self.recognition_predictor,
            self.detection_predictor,
            self.langs
        )
        self.text_chunker = TextChunker(max_tokens=450, overlap_tokens=50)

    def translate_text(self, text, source_lang_info, target_lang_info):
        """Translate text using chunking for long texts."""
        try:
            source_lang, target_lang = source_lang_info[1], target_lang_info[1]
            target_token = target_lang_info[2]
            
            translation_model = self.get_model(source_lang, target_lang)
            if not translation_model:
                return "Error: Could not load translation model"
            
            tokenizer = translation_model['tokenizer']
            model = translation_model['model']

            # Get chunks using TextChunker
            chunks = self.text_chunker.create_chunks(text)
            translated_chunks = []

            # Translate each chunk
            for chunk in chunks:
                try:
                    chunk_text = f"{target_token} {chunk.text}" if target_token else chunk.text
                    inputs = tokenizer(chunk_text, return_tensors="pt", padding=True)
                    translated = model.generate(**inputs)
                    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
                    translated_chunks.append(translated_text)
                except Exception as e:
                    logger.error(f"Error translating chunk {chunk.index}: {str(e)}")
                    translated_chunks.append(f"[Error translating part {chunk.index + 1}]")

            # Combine translations using TextChunker
            final_translation = self.text_chunker.combine_translations(text, chunks, translated_chunks)
            return re.sub(r'\s+', ' ', final_translation).strip()

        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return f"Translation error: {str(e)}"

    def get_model(self, source_lang, target_lang):
        """Get or load translation model."""
        model_key = f"helsinki-nmt-{source_lang}-{target_lang}"
        if model_key not in self.models:
            try:
                model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
                self.tokenizers[model_key] = MarianTokenizer.from_pretrained(model_name)
                self.models[model_key] = MarianMTModel.from_pretrained(model_name)
                logger.info(f"Loaded model: {model_key}")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                return None
        return {
            'tokenizer': self.tokenizers[model_key],
            'model': self.models[model_key]
        }

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
        
        source_lang_info = INDIVIDUAL_LANGUAGES[source_lang]
        
        # Handle language family translation
        if is_family:
            if family_name not in LANGUAGE_FAMILIES:
                raise ValueError(f"Invalid language family: {family_name}")
            if target_lang not in LANGUAGE_FAMILIES[family_name]:
                raise ValueError(f"Invalid target language {target_lang} for family {family_name}")
            target_lang_info = LANGUAGE_FAMILIES[family_name][target_lang]
        else:
            if target_lang not in INDIVIDUAL_LANGUAGES:
                raise ValueError(f"Invalid target language: {target_lang}")
            if target_lang == source_lang:
                raise ValueError("Source and target languages cannot be the same")
            target_lang_info = INDIVIDUAL_LANGUAGES[target_lang]

        translated_text = translation_service.translate_text(
            text,
            source_lang_info,
            target_lang_info
        )

        return jsonify({
            'success': True,
            'translated_text': translated_text
        })

    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        return jsonify({
            'success': False,
            'error': str(ve)
        })
    except Exception as e:
        logger.error(f"Translation failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': f"Translation failed: {str(e)}"
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
            source_lang_info = INDIVIDUAL_LANGUAGES[source_lang]
            if is_family:
                target_lang_info = LANGUAGE_FAMILIES[family_name][target_lang]
            else:
                target_lang_info = INDIVIDUAL_LANGUAGES[target_lang]

            translated_text = translation_service.translate_text(
                extracted_text,
                source_lang_info,
                target_lang_info
            )

            return jsonify({
                'success': True,
                'result': {
                    'extracted_text': extracted_text,
                    'translated_text': translated_text
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': "No text could be extracted from the document"
            })

    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(port=8080, debug=True)