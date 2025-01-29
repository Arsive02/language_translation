import logging

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from app.core.constants import LANGUAGE_FAMILIES, INDIVIDUAL_LANGUAGES

logger = logging.getLogger(__name__)

class TranslationService:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}

    def _get_model_key(self, source_lang: str, target_lang: str) -> str:
        return f"helsinki-nmt-{source_lang}-{target_lang}"

    def _get_language_info(self, lang_name: str, is_family: bool = False, family_name: str = None):
        if is_family and family_name:
            return LANGUAGE_FAMILIES[family_name][lang_name]
        return INDIVIDUAL_LANGUAGES[lang_name]

    def _load_model(self, source_lang: str, target_lang: str):
        model_key = self._get_model_key(source_lang, target_lang)
        
        if model_key not in self.models:
            try:
                model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
                self.tokenizers[model_key] = AutoTokenizer.from_pretrained(model_name)
                self.models[model_key] = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                logger.info(f"Loaded translation model: {model_key}")
            except Exception as e:
                logger.error(f"Error loading model {model_key}: {str(e)}")
                raise

    async def translate(self, text: str, source_lang_info: tuple, target_lang_info: tuple) -> str:
        try:
            source_lang, target_lang = source_lang_info[1], target_lang_info[1]
            target_token = target_lang_info[2]
            
            model_key = self._get_model_key(source_lang, target_lang)
            self._load_model(source_lang, target_lang)
            
            tokenizer = self.tokenizers[model_key]
            model = self.models[model_key]
            
            # Add language token if needed
            if target_token:
                text = f"{target_token} {text}"
            
            # Tokenize and translate
            inputs = tokenizer(text, return_tensors="pt", padding=True)
            translated = model.generate(**inputs)
            translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
            
            logger.info(f"Successfully translated text from {source_lang} to {target_lang}")
            return translated_text
            
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            raise