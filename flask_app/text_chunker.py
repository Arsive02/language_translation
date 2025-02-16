import re
import logging
import nltk

from typing import List, Optional
from dataclasses import dataclass
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

nltk.download('punkt_tab')

logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    """Class to represent a chunk of text with metadata"""
    text: str
    index: int
    token_count: int
    is_partial_sentence: bool = False
    original_start: int = 0
    original_end: int = 0

class TextChunker:
    """
    A utility class for chunking large texts into smaller pieces while preserving
    sentence boundaries and context where possible.
    """
    
    def __init__(
        self,
        max_tokens: int = 450,
        overlap_tokens: int = 50,
        preserve_paragraphs: bool = True
    ):
        """
        Initialize the TextChunker.
        
        Args:
            max_tokens: Maximum number of tokens per chunk
            overlap_tokens: Number of tokens to overlap between chunks
            preserve_paragraphs: Whether to try to preserve paragraph boundaries
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.preserve_paragraphs = preserve_paragraphs
        
    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text before chunking."""
        if not text:
            return ""
            
        # Replace multiple newlines with single \n
        text = re.sub(r'\n\s*\n', '\n', text)
        
        # Replace other whitespace characters with space
        text = re.sub(r'[\r\t\f\v]', ' ', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Clean up spaces around newlines
        text = re.sub(r' *\n *', '\n', text)
        
        # Remove spaces at the start and end of the text
        text = text.strip()
        
        # Handle bullet points and lists consistently
        text = re.sub(r'•\s*', '• ', text)
        text = re.sub(r'^\s*[-*]\s+', '• ', text, flags=re.MULTILINE)
        
        return text

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.
        This is a rough approximation - actual token count may vary by tokenizer.
        """
        # Split on whitespace and punctuation
        words = re.findall(r'\b\w+\b|[^\w\s]', text)
        return len(words)

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK."""
        try:
            return sent_tokenize(text)
        except Exception as e:
            logger.warning(f"Error in sentence tokenization: {e}")
            # Fallback to simple period-based splitting
            return [s.strip() + '.' for s in text.split('.') if s.strip()]

    def get_chunk_text(self, sentences: List[str], start_idx: int, max_tokens: int) -> tuple[str, int, bool]:
        """
        Get chunk text starting from start_idx that fits within max_tokens.
        Returns tuple of (chunk_text, end_idx, is_partial_sentence).
        """
        current_tokens = 0
        current_sentences = []
        is_partial = False
        
        for i in range(start_idx, len(sentences)):
            sentence = sentences[i]
            sentence_tokens = self.estimate_tokens(sentence)
            
            # If single sentence exceeds max tokens, split it
            if sentence_tokens > max_tokens:
                if not current_sentences:  # First sentence
                    words = sentence.split()
                    current_chunk = []
                    word_count = 0
                    
                    for word in words:
                        word_tokens = self.estimate_tokens(word)
                        if word_count + word_tokens <= max_tokens:
                            current_chunk.append(word)
                            word_count += word_tokens
                        else:
                            break
                            
                    chunk_text = ' '.join(current_chunk)
                    is_partial = True
                    return chunk_text, i, is_partial
                break
                
            # Check if adding this sentence would exceed the limit
            if current_tokens + sentence_tokens > max_tokens and current_sentences:
                break
                
            current_sentences.append(sentence)
            current_tokens += sentence_tokens
            
        return ' '.join(current_sentences), start_idx + len(current_sentences), is_partial

    def create_chunks(self, text: str) -> List[TextChunk]:
        """
        Split text into chunks that respect sentence boundaries where possible.
        
        Args:
            text: Input text to be chunked
            
        Returns:
            List of TextChunk objects
        """
        text = self.preprocess_text(text)
        if not text:
            return []
            
        chunks = []
        current_idx = 0
        
        # Split into paragraphs if preserve_paragraphs is True
        if self.preserve_paragraphs:
            paragraphs = text.split('\n')
        else:
            paragraphs = [text]
            
        # Add progress bar for paragraph processing
        for para in tqdm(paragraphs, desc="Processing paragraphs"):
            if not para.strip():
                continue
                
            sentences = self.split_into_sentences(para)
            para_start = 0
            
            while para_start < len(sentences):
                chunk_text, next_start, is_partial = self.get_chunk_text(
                    sentences, para_start, self.max_tokens
                )
                
                if not chunk_text:
                    break
                    
                # Calculate original text positions
                original_start = text.find(chunk_text)
                original_end = original_start + len(chunk_text)
                
                chunks.append(TextChunk(
                    text=chunk_text,
                    index=current_idx,
                    token_count=self.estimate_tokens(chunk_text),
                    is_partial_sentence=is_partial,
                    original_start=original_start,
                    original_end=original_end
                ))
                
                current_idx += 1
                para_start = next_start if not is_partial else next_start + 1
                
        return chunks

    def combine_translations(self, original_text: str, chunks: List[TextChunk], 
                           translations: List[str]) -> str:
        """
        Combine translated chunks back into a single text, handling overlaps.
        
        Args:
            original_text: Original input text
            chunks: List of TextChunk objects
            translations: List of translated text chunks
            
        Returns:
            Combined translated text
        """
        if len(chunks) != len(translations):
            raise ValueError("Number of chunks and translations must match")
            
        if len(chunks) == 0:
            return ""
            
        if len(chunks) == 1:
            return translations[0]
            
        # Combine translations, handling partial sentences
        result = []
        for i, (chunk, translation) in enumerate(zip(chunks, translations)):
            if i > 0 and chunk.is_partial_sentence:
                # For partial sentences, try to find a clean break point
                prev_translation = translations[i-1]
                overlap = self._find_overlap(prev_translation, translation)
                if overlap:
                    translation = translation[len(overlap):]
                    
            result.append(translation)
            
        return ' '.join(result)

    def _find_overlap(self, text1: str, text2: str, min_length: int = 10) -> Optional[str]:
        """Find overlapping text between two strings."""
        if not text1 or not text2:
            return None
            
        # Get the last part of text1 and first part of text2
        end_text = text1[-100:]  # Look at last 100 chars
        start_text = text2[:100]  # Look at first 100 chars
        
        # Find the longest common substring
        overlap = None
        for length in range(min(len(end_text), len(start_text)), min_length - 1, -1):
            if end_text[-length:] == start_text[:length]:
                overlap = start_text[:length]
                break
                
        return overlap