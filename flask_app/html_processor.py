import logging
from bs4 import BeautifulSoup, NavigableString, Tag
from typing import List, Tuple, Dict, Optional, Any

logger = logging.getLogger(__name__)

class HTMLProcessor:
    """
    An improved processor for HTML content that preserves exact HTML structure
    while only translating text content.
    """
    
    def __init__(self):
        self.skip_translation_class = 'notranslate'
        self.skip_tags = {
            'script', 'style', 'pre', 'code', 'head', 'title', 'meta',
            'link', 'iframe', 'noscript', 'svg', 'path', 'img'
        }
        
    def extract_text(self, html_content: str) -> Tuple[List[str], Dict[str, Any]]:
        """
        Extract translatable text nodes from HTML content while preserving exact structure.
        
        Args:
            html_content: HTML content as a string
            
        Returns:
            A tuple containing:
            - List of text fragments to translate
            - DOM map that maintains references to the exact nodes in the original structure
        """
        try:
            # Parse the HTML using 'html.parser' to ensure proper handling
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Use a list to store text fragments and their corresponding nodes
            text_fragments = []
            dom_map = {}
            
            # Process the soup to find all text nodes
            self._extract_text_from_node(soup, text_fragments, dom_map)
            
            return text_fragments, {'soup': soup, 'node_map': dom_map}
            
        except Exception as e:
            logger.error(f"Error extracting text from HTML: {str(e)}")
            return [], {}
    
    def _extract_text_from_node(self, node, text_fragments: List[str], dom_map: Dict[int, Any], path: str = ""):
        """
        Recursively extract text from nodes while maintaining exact structure.
        
        Args:
            node: The current BeautifulSoup node
            text_fragments: List to store extracted text
            dom_map: Dictionary to map indices to nodes
            path: Current path in the DOM tree for debugging
        """
        # Skip processing for certain tags
        if isinstance(node, Tag) and node.name in self.skip_tags:
            return
            
        # Skip elements with notranslate class
        if isinstance(node, Tag) and node.get('class') and self.skip_translation_class in node.get('class'):
            return
        
        # Process this node
        if isinstance(node, NavigableString) and node.parent and node.parent.name not in self.skip_tags:
            # Only process non-empty text
            text = str(node).strip()
            if text:
                index = len(text_fragments)
                text_fragments.append(text)
                dom_map[index] = node
        
        # Recursively process child nodes
        if isinstance(node, Tag):
            for child in node.children:
                child_path = f"{path}/{child.name}" if isinstance(child, Tag) else path
                self._extract_text_from_node(child, text_fragments, dom_map, child_path)
    
    def replace_text(self, dom_data: Dict[str, Any], translated_fragments: List[str]) -> str:
        """
        Replace the original text with translated text while keeping exact HTML structure.
        
        Args:
            dom_data: DOM data containing soup and node map
            translated_fragments: List of translated text fragments
            
        Returns:
            HTML content with translated text and preserved structure
        """
        try:
            soup = dom_data.get('soup')
            node_map = dom_data.get('node_map', {})
            
            if not soup or not node_map:
                logger.error("Invalid DOM data for text replacement")
                return ""
            
            # Replace text in each node
            for index, node in node_map.items():
                if index < len(translated_fragments):
                    # Replace the original string with the translated string
                    node.replace_with(NavigableString(translated_fragments[index]))
            
            # Return the HTML as a string
            return str(soup)
            
        except Exception as e:
            logger.error(f"Error replacing text in HTML: {str(e)}")
            return ""