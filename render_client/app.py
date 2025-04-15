import logging
import os
import time

from flask import Flask, g, jsonify, render_template, request

from api_client import TranslationClient
from lt_logger import TranslationLogger

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
translation_client = TranslationClient()

LANGUAGE_CODES = {
    # Common language codes (using 2-letter ISO codes)
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
    'Tamil': 'tam',
    'Telugu': 'tel',
    'Kannada': 'kan',
    'Malayalam': 'mal',
    'Czech': 'cs',
    'Slovak': 'sk',
    'Swedish': 'sv',
    'Danish': 'da'
}

SPECIAL_TOKEN_LANGUAGES = {
    'Tamil', 'Telugu', 'Kannada', 'Malayalam'
}

DRAVIDIAN_MODEL = "Helsinki-NLP/opus-mt-en-dra"

@app.before_request
def before_request():
    g.start_time = time.time()

@app.after_request
def after_request(response):
    if hasattr(g, 'start_time'):
        duration = time.time() - g.start_time
        logger.info(f"Request to {request.path} took {duration:.2f}s")
    return response

@app.route('/')
def index():
    return render_template(
        'index.html',
        languages=LANGUAGE_CODES
    )

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for monitoring"""
    try:
        api_health = translation_client.health_check()
        
        status = {
            "status": "ok",
            "api_status": api_health.get("status", "unknown"),
            "timestamp": time.time()
        }
        
        return jsonify(status)
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "timestamp": time.time()
        })

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.form
        text = data.get('text', '')
        source_lang = data.get('source_lang', 'English')
        target_lang = data.get('target_lang', '')

        logger.info(f"Text translation request: {len(text)} chars from {source_lang} to {target_lang}")

        if source_lang not in LANGUAGE_CODES:
            raise ValueError(f"Invalid source language: {source_lang}")
        
        if target_lang not in LANGUAGE_CODES:
            raise ValueError(f"Invalid target language: {target_lang}")
            
        if target_lang == source_lang:
            raise ValueError("Source and target languages cannot be the same")
            
        source_lang_code = LANGUAGE_CODES[source_lang]
        target_lang_code = LANGUAGE_CODES[target_lang]
        
        if target_lang in SPECIAL_TOKEN_LANGUAGES:
            text = f">>{target_lang_code}<<{text}"
            target_lang_code = "dra"

        timeout = 30 + min(len(text) // 1000, 60)  # 30s base + 1s per 1000 chars, max 90s
        
        response = translation_client.translate_text(
            text,
            source_lang_code,
            target_lang_code,
            timeout=timeout
        )

        translated_text = response.get('translated_text', '')

        translation_logger.log_translation(
            source_text=text,
            translated_text=translated_text,
            source_lang=source_lang,
            target_lang=target_lang,
            is_family=False,
            family_name="",
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
            is_family=False,
            family_name="",
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

        logger.info(f"HTML translation request: {len(html_content)} chars from {source_lang} to {target_lang}")

        if source_lang not in LANGUAGE_CODES:
            raise ValueError(f"Invalid source language: {source_lang}")
        
        if target_lang not in LANGUAGE_CODES:
            raise ValueError(f"Invalid target language: {target_lang}")
            
        if target_lang == source_lang:
            raise ValueError("Source and target languages cannot be the same")
            
        source_lang_code = LANGUAGE_CODES[source_lang]
        target_lang_code = LANGUAGE_CODES[target_lang]
        
        special_token = ""
        if target_lang in SPECIAL_TOKEN_LANGUAGES:
            special_token = f">>{target_lang_code}<<"
            target_lang_code = "dra"

        timeout = 60 + min(len(html_content) // 500, 120)  # 60s base + 1s per 500 chars, max 180s

        response = translation_client.translate_html(
            html_content,
            source_lang_code,
            target_lang_code,
            special_token=special_token,
            timeout=timeout
        )

        translated_html = response.get('translated_html', '')

        translation_logger.log_translation(
            source_text=html_content,
            translated_text=translated_html,
            source_lang=source_lang,
            target_lang=target_lang,
            is_family=False,
            family_name="",
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
            is_family=False,
            family_name="",
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

        logger.info(f"Document processing request: {file.filename}, {source_lang} to {target_lang}, OCR: {use_ocr}")

        if source_lang not in LANGUAGE_CODES:
            raise ValueError(f"Invalid source language: {source_lang}")
            
        if target_lang not in LANGUAGE_CODES:
            raise ValueError(f"Invalid target language: {target_lang}")
            
        if target_lang == source_lang:
            raise ValueError("Source and target languages cannot be the same")
            
        source_lang_code = LANGUAGE_CODES[source_lang]
        target_lang_code = LANGUAGE_CODES[target_lang]
        
        special_token = ""
        if target_lang in SPECIAL_TOKEN_LANGUAGES:
            special_token = f">>{target_lang_code}<<"
            target_lang_code = "dra"

        file_data = file.read()
        
        file_size_mb = len(file_data) / (1024 * 1024)
        timeout = 120 + min(int(file_size_mb * 30), 240)  # 120s base + 30s per MB, max 360s
        
        response = translation_client.process_document(
            file_data,
            file.filename,
            source_lang_code,
            target_lang_code,
            special_token=special_token,
            use_ocr=use_ocr,
            timeout=timeout
        )

        extracted_text = response.get('extracted_text', '')
        translated_text = response.get('translated_text', '')

        translation_logger.log_translation(
            source_text=extracted_text,
            translated_text=translated_text,
            source_lang=source_lang,
            target_lang=target_lang,
            is_family=False,
            family_name="",
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

    except Exception as e:
        error_message = str(e)
        logger.error(f"Document processing error: {error_message}")
        
        translation_logger.log_translation(
            source_text="",
            translated_text="",
            source_lang=request.form.get('source_lang', 'English'),
            target_lang=request.form.get('target_lang', ''),
            is_family=False,
            family_name="",
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
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8000)), debug=False)