from typing import List, Dict, Any
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
from collections import Counter

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class ResponseEvaluator:
    """Evaluates chatbot responses using various metrics"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def calculate_coherence(self, response: str) -> float:
        """Calculate coherence score based on sentence structure and flow"""
        # Split response into sentences
        sentences = sent_tokenize(response)
        if len(sentences) < 2:
            return 1.0  # Single sentence is considered coherent
        
        # Calculate word overlap between consecutive sentences
        coherence_scores = []
        for i in range(len(sentences) - 1):
            words1 = set(word_tokenize(sentences[i].lower()))
            words2 = set(word_tokenize(sentences[i + 1].lower()))
            
            # Calculate Jaccard similarity
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            similarity = intersection / union if union > 0 else 0
            coherence_scores.append(similarity)
        
        return np.mean(coherence_scores)
    
    def calculate_relevance(self, query: str, response: str) -> float:
        """Calculate relevance score based on query-response similarity"""
        # Tokenize and normalize
        query_words = set(word_tokenize(query.lower()))
        response_words = set(word_tokenize(response.lower()))
        
        # Remove stop words
        query_words = query_words - self.stop_words
        response_words = response_words - self.stop_words
        
        # Calculate word overlap
        common_words = query_words.intersection(response_words)
        total_query_words = len(query_words)
        
        if total_query_words == 0:
            return 0.0
        
        return len(common_words) / total_query_words
    
    def calculate_helpfulness(self, response: str) -> float:
        """Calculate helpfulness score based on response characteristics"""
        # Split response into sentences
        sentences = sent_tokenize(response)
        
        # Calculate various helpfulness indicators
        indicators = {
            'length': min(len(response.split()) / 100, 1.0),  # Length score
            'structure': self._calculate_structure_score(sentences),  # Structure score
            'information_density': self._calculate_information_density(response),  # Information density
            'clarity': self._calculate_clarity_score(response)  # Clarity score
        }
        
        # Weighted average of indicators
        weights = {
            'length': 0.2,
            'structure': 0.3,
            'information_density': 0.3,
            'clarity': 0.2
        }
        
        helpfulness_score = sum(
            score * weights[indicator]
            for indicator, score in indicators.items()
        )
        
        return helpfulness_score
    
    def _calculate_structure_score(self, sentences: List[str]) -> float:
        """Calculate structure score based on sentence organization"""
        if not sentences:
            return 0.0
        
        # Check for transition words
        transition_words = {
            'first', 'second', 'third', 'finally', 'however',
            'therefore', 'thus', 'consequently', 'furthermore',
            'moreover', 'additionally', 'in conclusion'
        }
        
        # Count sentences with transition words
        structured_sentences = sum(
            1 for sent in sentences
            if any(word in sent.lower() for word in transition_words)
        )
        
        return structured_sentences / len(sentences)
    
    def _calculate_information_density(self, text: str) -> float:
        """Calculate information density score"""
        # Remove stop words and punctuation
        words = re.findall(r'\w+', text.lower())
        words = [w for w in words if w not in self.stop_words]
        
        if not words:
            return 0.0
        
        # Calculate unique word ratio
        unique_words = len(set(words))
        total_words = len(words)
        
        return unique_words / total_words
    
    def _calculate_clarity_score(self, text: str) -> float:
        """Calculate clarity score based on readability and complexity"""
        # Simple readability metrics
        sentences = sent_tokenize(text)
        words = text.split()
        
        if not sentences or not words:
            return 0.0
        
        # Average sentence length (normalized)
        avg_sentence_length = len(words) / len(sentences)
        sentence_length_score = 1.0 - min(avg_sentence_length / 30, 1.0)
        
        # Word complexity (ratio of long words)
        long_words = sum(1 for word in words if len(word) > 6)
        complexity_score = 1.0 - (long_words / len(words))
        
        return (sentence_length_score + complexity_score) / 2
    
    def evaluate_response(self, query: str, response: str) -> Dict[str, float]:
        """Evaluate a response using all metrics"""
        return {
            'coherence': self.calculate_coherence(response),
            'relevance': self.calculate_relevance(query, response),
            'helpfulness': self.calculate_helpfulness(response)
        }
