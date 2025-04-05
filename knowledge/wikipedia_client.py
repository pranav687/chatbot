import wikipediaapi
from typing import List, Dict, Any
import re

from utils.config import WIKIPEDIA_USER_AGENT


class WikipediaClient:
    """Client for retrieving information from Wikipedia"""
    
    def __init__(self):
        """Initialize the Wikipedia API client"""
        self.wiki = wikipediaapi.Wikipedia(
            language='en',
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent=WIKIPEDIA_USER_AGENT
        )
    
    def search(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Search Wikipedia for a given query
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries containing page titles and summaries
        """
        # For each word in the query, try to get a Wikipedia page
        words = re.findall(r'\b\w+\b', query)
        combined_terms = []
        
        # Add individual words
        for word in words:
            if len(word) > 3:  # Only consider words with more than 3 characters
                combined_terms.append(word)
        
        # Add pairs of consecutive words
        for i in range(len(words) - 1):
            if len(words[i]) > 1 and len(words[i+1]) > 1:
                combined_terms.append(f"{words[i]} {words[i+1]}")
        
        # Add the full query
        combined_terms.append(query)
        
        # Remove duplicates
        combined_terms = list(set(combined_terms))
        
        # Search for each term
        results = []
        for term in combined_terms:
            page = self.wiki.page(term)
            if page.exists():
                results.append({
                    "title": page.title,
                    "summary": page.summary[0:500],  # First 500 chars of summary
                    "url": page.fullurl,
                    "content": page.text[0:5000]  # First 5000 chars of content
                })
                
                # If we have enough results, stop
                if len(results) >= limit:
                    break
        
        return results[:limit]
    
    def get_page(self, title: str) -> Dict[str, Any]:
        """
        Get a Wikipedia page by title
        
        Args:
            title: Page title
            
        Returns:
            Dictionary containing page information
        """
        page = self.wiki.page(title)
        if page.exists():
            return {
                "title": page.title,
                "summary": page.summary,
                "url": page.fullurl,
                "content": page.text,
                "sections": [
                    {
                        "title": section.title,
                        "level": section.level,
                        "text": section.text
                    }
                    for section in page.sections
                ]
            }
        else:
            return None
    
    def get_section(self, title: str, section_name: str) -> str:
        """
        Get a specific section from a Wikipedia page
        
        Args:
            title: Page title
            section_name: Section name
            
        Returns:
            Section text or None if not found
        """
        page = self.wiki.page(title)
        if not page.exists():
            return None
            
        # Find the section
        for section in page.sections:
            if section.title.lower() == section_name.lower():
                return section.text
        
        return None