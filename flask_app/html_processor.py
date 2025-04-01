import logging
from bs4 import BeautifulSoup
from typing import List, Tuple, Dict, Optional, Any

logger = logging.getLogger(__name__)

class HTMLProcessor:
    """
    A processor for HTML content that extracts text for translation
    while preserving the HTML structure.
    """
    
    def __init__(self):
        self.translatable_tags = [
            'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 
            'li', 'td', 'th', 'caption', 'span', 'div', 
            'a', 'button', 'label', 'title'
        ]
        self.skip_translation_class = 'notranslate'
        
    def extract_text(self, html_content: str) -> Tuple[List[str], Dict[int, Any]]:
        """
        Extract text from HTML content for translation.
        
        Args:
            html_content: HTML content as a string
            
        Returns:
            A tuple containing:
            - List of text fragments to translate
            - Map of HTML elements and their indices in the text fragments list
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find all elements that contain text to translate
            elements = self._find_translatable_elements(soup)
            
            text_fragments = []
            html_map = {}
            
            for i, element in enumerate(elements):
                # Skip elements with the notranslate class
                if element.get('class') and self.skip_translation_class in element.get('class'):
                    continue
                    
                # Get text content
                text = element.get_text().strip()
                if text:
                    text_fragments.append(text)
                    html_map[len(text_fragments) - 1] = element
            
            return text_fragments, html_map
            
        except Exception as e:
            logger.error(f"Error extracting text from HTML: {str(e)}")
            return [], {}
    
    def _find_translatable_elements(self, soup: BeautifulSoup) -> List[Any]:
        """
        Find all HTML elements that should be translated.
        
        Args:
            soup: BeautifulSoup object of HTML content
            
        Returns:
            List of HTML elements to translate
        """
        elements = []
        
        for tag_name in self.translatable_tags:
            # Find all elements of this tag type
            tags = soup.find_all(tag_name)
            
            for tag in tags:
                # Skip empty tags
                if not tag.get_text().strip():
                    continue
                    
                # Skip tags that are children of already selected tags
                if self._is_child_of_selected(tag, elements):
                    continue
                    
                # Skip tags with notranslate class
                if tag.get('class') and self.skip_translation_class in tag.get('class'):
                    continue
                    
                elements.append(tag)
        
        return elements
    
    def _is_child_of_selected(self, tag: Any, selected_elements: List[Any]) -> bool:
        """
        Check if a tag is a child of any of the already selected elements.
        
        Args:
            tag: The tag to check
            selected_elements: List of already selected elements
            
        Returns:
            True if tag is a child of any selected element, False otherwise
        """
        for parent in tag.parents:
            if parent in selected_elements:
                return True
        return False
    
    def replace_text(self, html_map: Dict[int, Any], translated_fragments: List[str]) -> str:
        """
        Replace the original text with translated text in the HTML structure.
        
        Args:
            html_map: Map of HTML elements and their indices
            translated_fragments: List of translated text fragments
            
        Returns:
            HTML content with translated text
        """
        try:
            # Make a copy of the BeautifulSoup object from the first element
            if not html_map or not translated_fragments:
                return ""
                
            first_element = next(iter(html_map.values()))
            soup = first_element.find_parent()
            while soup.parent:
                soup = soup.parent
                
            # Replace text in each element
            for index, element in html_map.items():
                if index < len(translated_fragments):
                    # Replace the text content of the element
                    element.string = translated_fragments[index]
            
            return str(soup)
            
        except Exception as e:
            logger.error(f"Error replacing text in HTML: {str(e)}")
            return ""