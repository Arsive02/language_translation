import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
from PIL import Image
from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
import json
import base64

def load_examples():
    with open("annotations/claude_data_extraction.json", "r") as f:
        examples = json.load(f)
    return examples

# Sample images for demo
SAMPLE_IMAGES = [
    "data/images_with_text/IMG-20250117-WA0006.jpg",
    "data/images_with_text/IMG-20250117-WA0007.jpg",
    "data/images_with_text/IMG-20250117-WA0008.jpg",
    "data/images_with_text/IMG-20250117-WA0009.jpg",
    "data/images_with_text/IMG-20250117-WA0012.jpg",
    "data/images_with_text/IMG-20250117-WA0015.jpg",
    "data/images_with_text/IMG-20250117-WA0017.jpg",
    "data/images_with_text/IMG-20250117-WA0018.jpg",
    "data/images_with_text/IMG-20250117-WA0022.jpg",
    "data/images_with_text/IMG-20250117-WA0025.jpg",
    "data/images_with_text/IMG-20250117-WA0026.jpg",
    "data/images_with_text/IMG-20250117-WA0029.jpg",
    "data/images_with_text/IMG-20250117-WA0031.jpg",
    "data/images_with_text/IMG-20250117-WA0035.jpg",
    "data/images_with_text/IMG-20250117-WA0036.jpg",
    "data/images_with_text/IMG-20250117-WA0040.jpg",
    "data/images_with_text/IMG-20250117-WA0041.jpg",
    "data/images_with_text/IMG-20250117-WA0044.jpg",
]

langs = ["en"]  # OCR language detection

# Language families and their members
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
        'Spanish': ('en', 'spa', '>>spa<<'),
        'French': ('en', 'fra', '>>fra<<'),
        'Italian': ('en', 'ita', '>>ita<<'),
        'Portuguese': ('en', 'por', '>>por<<')
    }
}

# Available individual languages
INDIVIDUAL_LANGUAGES = {
    'English': ('en', 'en', '>>eng<<'),
    'Spanish': ('en', 'spa', '>>spa<<'),
    'French': ('en', 'fra', '>>fra<<'),
    'German': ('en', 'deu', '>>deu<<'),
    'Italian': ('en', 'ita', '>>ita<<'),
    'Portuguese': ('en', 'por', '>>por<<'),
    'Dutch': ('en', 'nld', '>>nld<<'),
    'Polish': ('en', 'pol', '>>pol<<'),
    'Russian': ('en', 'rus', '>>rus<<')
}

# Page config
st.set_page_config(
    page_title="Universal Translator",
    page_icon="🌎",
    layout="wide"
)

# Initialize session state
if 'translation_models' not in st.session_state:
    st.session_state.translation_models = {}
if 'examples' not in st.session_state:
    try:
        st.session_state.examples = load_examples()
    except:
        st.session_state.examples = {}

# Initialize OCR model
@st.cache_resource
def load_ocr_model():
    det_processor, det_model = load_det_processor(), load_det_model()
    rec_model, rec_processor = load_rec_model(), load_rec_processor()
    return det_processor, det_model, rec_processor, rec_model

@st.cache_resource
def get_translation_model(source_lang, target_lang):
    """Get or load translation model based on language pair"""
    model_key = f"helsinki-nmt-{source_lang}-{target_lang}"
    
    if model_key not in st.session_state.translation_models:
        model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
        try:
            st.session_state.translation_models[model_key] = {
                'tokenizer': MarianTokenizer.from_pretrained(model_name),
                'model': MarianMTModel.from_pretrained(model_name)
            }
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
    
    return st.session_state.translation_models.get(model_key)

def translate_text(text, source_lang_info, target_lang_info):
    """Translate text using the model and appropriate tokens"""
    try:
        source_lang, target_lang = source_lang_info[1], target_lang_info[1]
        target_token = target_lang_info[2]
        
        # Get model and tokenizer
        translation = get_translation_model(source_lang, target_lang)
        if not translation:
            return "Error: Could not load translation model"
            
        tokenizer = translation['tokenizer']
        model = translation['model']
        
        # Add language token if needed
        if target_token:
            text = f"{target_token} {text}"
        
        # Tokenize and translate
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        translated = model.generate(**inputs)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        
        return translated_text
    
    except Exception as e:
        return f"Translation error: {str(e)}"

def main():
    st.title("🌎 Universal Translator")
    st.write("Translate text and images using state-of-the-art models")
    
    # Sidebar for language selection
    with st.sidebar:
        st.header("Language Selection")
        source_lang = 'English'  # Fixed source language
        source_lang_info = INDIVIDUAL_LANGUAGES[source_lang]
        
        # Target language selection
        target_type = st.radio("Target Language Type", ["Individual", "Language Family"])
        
        if target_type == "Individual":
            target_lang = st.selectbox(
                "Target Language",
                options=list(INDIVIDUAL_LANGUAGES.keys()),
                key="target_individual"
            )
            target_lang_info = INDIVIDUAL_LANGUAGES[target_lang]
        else:
            target_family = st.selectbox(
                "Target Language Family",
                options=list(LANGUAGE_FAMILIES.keys()),
                key="target_family"
            )
            target_lang = st.selectbox(
                "Specific Target Language",
                options=list(LANGUAGE_FAMILIES[target_family].keys()),
                key="target_specific"
            )
            target_lang_info = LANGUAGE_FAMILIES[target_family][target_lang]
    
    # Main content area
    tab1, tab2 = st.tabs(["Text Translation", "Image Translation"])
    
    # Text Translation Tab
    with tab1:
        st.subheader("Text Translation")
        
        # Example selection
        use_example = st.checkbox("Use example text")
        
        if use_example and st.session_state.examples:
            examples = st.session_state.examples.get('exhibits', [])
            example_titles = [f"{exhibit['title']}" for exhibit in examples]
            selected_example = st.selectbox(
                "Select an example text",
                range(len(example_titles)),
                format_func=lambda x: example_titles[x]
            )
            input_text = examples[selected_example]['description']
            st.text_area("Selected example text", value=input_text, height=150, disabled=True)
        else:
            input_text = st.text_area("Enter text to translate", height=150)
        
        if st.button("Translate Text", key="translate_text"):
            if input_text:
                with st.spinner("Translating..."):
                    translated = translate_text(input_text, source_lang_info, target_lang_info)
                    st.success("Translation Complete!")
                    st.write("### Translated Text:")
                    st.write(translated)
            else:
                st.warning("Please enter some text to translate")
    
    # Image Translation Tab
    with tab2:
        try:
            st.subheader("Image Translation")
            
            # Image source selection
            image_source = st.radio("Image Source", ["Upload Image", "Use Sample Image"])
            
            if image_source == "Upload Image":
                uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
            else:
                selected_sample = st.selectbox("Select sample image", SAMPLE_IMAGES)
                image = Image.open(selected_sample)
            
            if 'image' in locals():  # If an image is loaded (either uploaded or sample)
                st.image(image, caption="Selected Image", use_container_width=True)
                
                if st.button("Extract and Translate Text", key="translate_image"):
                    with st.spinner("Processing image..."):
                        # Extract text using OCR
                        det_processor, det_model, rec_processor, rec_model = load_ocr_model()
                        predictions = run_ocr([image], [langs], det_model, det_processor, rec_model, rec_processor)
                        extracted_text = " ".join([line.text for line in predictions[0].text_lines])
                        
                        if extracted_text:
                            st.write("### Extracted Text:")
                            st.write(extracted_text)
                            st.session_state.source_text = extracted_text
                            
                            st.write("### Translated Text:")
                            translated = translate_text(extracted_text, source_lang_info, target_lang_info)
                            st.write(translated)
                            st.session_state.translated_text = translated
                        else:
                            st.warning("No text detected in the image")

                            
                def get_download_link(data, filename):
                    """Generate a link to download data as a file."""
                    json_data = json.dumps(data)
                    b64 = base64.b64encode(json_data.encode()).decode()
                    return f'<a href="data:application/json;base64,{b64}" download="{filename}">Download JSON file</a>'

                if 'translated_text' not in st.session_state:
                    st.session_state.translated_text = ""
                if 'source_text' not in st.session_state:
                    st.session_state.source_text = ""

                if st.button("Save Translation", key="save_translation"):
                    if st.session_state.translated_text and st.session_state.source_text:
                        data = {
                            "source_text": st.session_state.source_text,
                            "translated_text": st.session_state.translated_text
                        }
                        download_link = get_download_link(data, "translated_text.json")
                        st.markdown(download_link, unsafe_allow_html=True)
                    else:
                        st.warning("No translated text to save")
        except Exception as e:
            st.error(f"Seems like there is a problem loading OCR model: {str(e)}. I am working on it.")
if __name__ == "__main__":
    main()