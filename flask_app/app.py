from flask import Flask, render_template, request, jsonify
from transformers import MarianMTModel, MarianTokenizer
from PIL import Image
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
import logging
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

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
        'Russian': ('en', 'rus', '>>rus<<'),
        'Polish': ('en', 'pol', '>>pol<<'),
        'Czech': ('en', 'ces', '>>ces<<'),
        'Slovak': ('en', 'slk', '>>slk<<')
    },
    'Germanic': {
        'German': ('en', 'deu', '>>deu<<'),
        'Dutch': ('en', 'nld', '>>nld<<'),
        'Swedish': ('en', 'swe', '>>swe<<'),
        'Danish': ('en', 'dan', '>>dan<<')
    },
    'Romance': {
        'Spanish': ('eng', 'spa', '>>spa<<'),
        'French': ('eng', 'fra', '>>fra<<'),
        'Italian': ('eng', 'ita', '>>ita<<'),
        'Portuguese': ('eng', 'por', '>>por<<')
    }
}

INDIVIDUAL_LANGUAGES = {
    'English': ('en', 'en', '>>eng<<'),
    'Spanish': ('en', 'es', '>>spa<<'),
    'French': ('eng', 'fra', '>>fra<<'),
    'German': ('en', 'deu', '>>deu<<'),
    'Italian': ('en', 'ita', '>>ita<<'),
    'Portuguese': ('en', 'por', '>>por<<'),
    'Dutch': ('en', 'nld', '>>nld<<'),
    'Polish': ('en', 'pol', '>>pol<<'),
    'Russian': ('en', 'rus', '>>rus<<')
}

# Initialize OCR and translation models
class TranslationService:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.recognition_predictor = RecognitionPredictor()
        self.detection_predictor = DetectionPredictor()
        self.langs = ["en"]

    def get_model(self, source_lang, target_lang):
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

    def translate_text(self, text, source_lang_info, target_lang_info):
        try:
            source_lang, target_lang = source_lang_info[1], target_lang_info[1]
            target_token = target_lang_info[2]
            
            translation = self.get_model(source_lang, target_lang)
            if not translation:
                return "Error: Could not load translation model"
                
            tokenizer = translation['tokenizer']
            model = translation['model']
            
            if target_token:
                text = f"{target_token} {text}"
            
            inputs = tokenizer(text, return_tensors="pt", padding=True)
            translated = model.generate(**inputs)
            translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
            
            return translated_text
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return f"Translation error: {str(e)}"

    def process_image(self, image_path):
        try:
            image = Image.open(image_path)
            predictions = self.recognition_predictor(
                [image], 
                [self.langs], 
                self.detection_predictor
            )
            
            if predictions and len(predictions) > 0:
                # Get all text lines from the first page
                extracted_text = " ".join(
                    line.text for line in predictions[0].text_lines
                    if hasattr(line, 'text') and line.text.strip()
                )
                return extracted_text.strip()
            return ""
        
        except Exception as e:
            logger.error(f"OCR error: {str(e)}")
            return ""

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
        is_family = data.get('is_family') == 'true'
        family_name = data.get('family_name', '')

        source_lang_info = INDIVIDUAL_LANGUAGES[source_lang]
        if is_family:
            target_lang_info = LANGUAGE_FAMILIES[family_name][target_lang]
        else:
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

    except Exception as e:
        logger.error(f"Translation failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/process-document', methods=['POST'])
def process_document():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Extract text from image
            extracted_text = translation_service.process_image(filepath)
            
            # Clean up uploaded file
            os.remove(filepath)

            if not extracted_text:
                return jsonify({
                    'success': False,
                    'error': 'No text detected in the image'
                })

            # Translate the extracted text
            source_lang = request.form.get('source_lang', 'English')
            target_lang = request.form.get('target_lang', '')
            is_family = request.form.get('is_family') == 'true'
            family_name = request.form.get('family_name', '')

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
                'extracted_text': extracted_text,
                'translated_text': translated_text
            })

    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)