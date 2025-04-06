from typing import List, Dict, Tuple
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from scipy import stats
import pandas as pd
from datetime import datetime
import json
import os
import logging
from functools import lru_cache
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResponseEvaluator:
    """Comprehensive chatbot evaluator with user satisfaction metrics"""
    
    def __init__(self, enable_llm_judge: bool = True):
        """
        Args:
            enable_llm_judge: Whether to use semantic similarity for accuracy checks
        """
        self._initialize_nltk()
        self.stop_words = set(stopwords.words('english'))
        self.enable_llm_judge = enable_llm_judge
        
        # Initialize models
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            if enable_llm_judge:
                self.nli_model = self.sentence_model  # Reuse model
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
            
        # Initialize evaluation log
        os.makedirs("eval_logs", exist_ok=True)
        self.log_file = f"eval_logs/{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl"
        
        # Configure metrics weights
        self.weights = {
            'accuracy': 0.3,
            'coherence': 0.2,
            'relevance': 0.2,
            'helpfulness': 0.2,
            'user_satisfaction': 0.1
        }
        
        # User satisfaction components
        self.satisfaction_factors = {
            'politeness': 0.3,
            'engagement': 0.2,
            'emotional_tone': 0.2,
            'personalization': 0.3
        }
    
    def _initialize_nltk(self):
        """Ensure all required NLTK resources are available"""
        required_data = [
            ('tokenizers/punkt', 'punkt'),
            ('corpora/stopwords', 'stopwords'),
            ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')
        ]
        
        for path, package in required_data:
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(package, quiet=True)
    
    # ========================
    # Core Evaluation Pipeline
    # ========================
    
    def evaluate_response(self, query: str, response: str, session_id: str = None) -> Dict[str, float]:
        """
        Complete evaluation pipeline with user satisfaction
        
        Returns:
            {
                'accuracy': 0-1,
                'coherence': 0-1,
                'relevance': 0-1,
                'helpfulness': 0-1,
                'user_satisfaction': 0-1,
                'composite_score': weighted average
            }
        """
        metrics = {
            'accuracy': self._calculate_accuracy(response),
            'coherence': self._calculate_coherence(response),
            'relevance': self._calculate_relevance(query, response),
            'helpfulness': self._calculate_helpfulness(query, response),
            'user_satisfaction': self._calculate_user_satisfaction(query, response)
        }
        
        metrics['composite_score'] = self._calculate_composite_score(metrics)
        # self._log_evaluation(session_id, query, response, metrics)
        
        return metrics
    
    def _calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """Calculate weighted composite score"""
        return sum(metrics[k] * self.weights[k] for k in metrics)
    
    # ========================
    # User Satisfaction Metric
    # ========================
    
    def _calculate_user_satisfaction(self, query: str, response: str) -> float:
        """
        Comprehensive user satisfaction metric combining:
        1. Politeness markers
        2. Engagement level
        3. Emotional tone
        4. Personalization
        """
        # Tokenize and normalize
        response_lower = response.lower()
        words = set(word_tokenize(response_lower))
        
        # 1. Politeness analysis
        polite_phrases = {
            'please', 'thank you', 'you\'re welcome', 'happy to help',
            'kindly', 'appreciate', 'welcome', 'my pleasure'
        }
        politeness = 1.0 if any(phrase in response_lower for phrase in polite_phrases) else 0.5
        
        # 2. Engagement analysis
        engagement_features = {
            'question_asking': 0.3 if '?' in response else 0,
            'suggestion_making': 0.2 if any(w in words for w in {'can', 'could', 'might', 'consider'}) else 0,
            'interactivity': 0.1 if any(w in words for w in {'you', 'your'}) else 0
        }
        engagement = sum(engagement_features.values())
        
        # 3. Emotional tone analysis
        positive_words = {
            'great', 'excellent', 'wonderful', 'happy', 'glad', 'pleased',
            'delighted', 'perfect', 'awesome', 'fantastic'
        }
        negative_words = {
            'sorry', 'unfortunately', 'regret', 'apologize', 'cannot',
            'unable', 'limit', 'restrict'
        }
        positive_count = len(words & positive_words)
        negative_count = len(words & negative_words)
        
        if positive_count > negative_count:
            emotional_tone = 0.8
        elif negative_count > positive_count:
            emotional_tone = 0.3
        else:
            emotional_tone = 0.6
        
        # 4. Personalization score
        query_terms = set(word_tokenize(query.lower())) - self.stop_words
        personal_terms = {'you', 'your'} | query_terms
        personalization = min(1.0, len(words & personal_terms) / 3)
        
        # Combine components
        satisfaction_score = (
            self.satisfaction_factors['politeness'] * politeness +
            self.satisfaction_factors['engagement'] * engagement +
            self.satisfaction_factors['emotional_tone'] * emotional_tone +
            self.satisfaction_factors['personalization'] * personalization
        )
        
        return min(max(satisfaction_score, 0), 1)  # Clamp to 0-1 range
    
    # ========================
    # Other Metric Implementations
    # ========================
    
    def _calculate_accuracy(self, response: str) -> float:
        """Accuracy assessment with self-consistency and factual checks"""
        sentences = sent_tokenize(response)
        if not sentences:
            return 0.0
            
        if len(sentences) == 1:
            return self._estimate_factual_confidence(sentences[0])
        
        consistency_score = self._check_self_consistency(sentences)
        factual_scores = [self._estimate_factual_confidence(s) for s in sentences]
        
        return (0.6 * np.mean(factual_scores)) + (0.4 * consistency_score)
    
    def _calculate_coherence(self, response: str) -> float:
        """Coherence metric evaluating flow and structure"""
        sentences = sent_tokenize(response)
        if len(sentences) < 2:
            return 1.0 if sentences else 0.0
        
        embeddings = self.sentence_model.encode(sentences)
        transition_scores = []
        for i in range(len(sentences)-1):
            sim = util.cos_sim(embeddings[i], embeddings[i+1]).item()
            transition_scores.append(sim)
        
        return np.mean(transition_scores)
    
    def _calculate_relevance(self, query: str, response: str) -> float:
        """Relevance metric combining semantic and lexical similarity"""
        query_emb = self.sentence_model.encode(query)
        response_emb = self.sentence_model.encode(response)
        semantic = util.cos_sim(query_emb, response_emb).item()
        
        query_words = set(word_tokenize(query.lower())) - self.stop_words
        response_words = set(word_tokenize(response.lower()))
        lexical = len(query_words & response_words) / max(1, len(query_words))
        
        return (0.7 * semantic) + (0.3 * lexical)
    
    def _calculate_helpfulness(self, query: str, response: str) -> float:
        """Helpfulness metric assessing practical utility"""
        # Completeness
        query_terms = set(word_tokenize(query.lower())) - self.stop_words
        response_terms = set(word_tokenize(response.lower()))
        completeness = len(query_terms & response_terms) / max(1, len(query_terms))
        
        # Actionability
        action_words = {'should', 'can', 'could', 'recommend', 'suggest', 'try', 'use'}
        action_score = 0.3 if any(word in response.lower() for word in action_words) else 0
        
        # Clarity
        clarity = self._calculate_clarity(response)
        
        return min(completeness + action_score + clarity, 1.0)
    
    # ========================
    # Helper Methods
    # ========================
    
    def _check_self_consistency(self, sentences: List[str]) -> float:
        """Check for contradictions using semantic similarity"""
        if not self.enable_llm_judge or len(sentences) < 2:
            return 1.0
            
        embeddings = self.nli_model.encode(sentences)
        sim_matrix = util.cos_sim(embeddings, embeddings).numpy()
        np.fill_diagonal(sim_matrix, -1)
        return max(0, min(1, (np.min(sim_matrix) + 0.7) / 0.7))
    
    def _estimate_factual_confidence(self, text: str) -> float:
        """Estimate confidence in factual claims"""
        claims = self._extract_claims(text)
        if not claims:
            return 1.0
            
        scores = []
        for claim in claims:
            tokens = set(word_tokenize(claim.lower()))
            if tokens & {'may', 'might', 'could', 'possibly'}:
                scores.append(0.4)
            elif any(marker in claim for marker in ['according to', 'studies show']):
                scores.append(0.9)
            else:
                scores.append(0.7)
        return np.mean(scores)
    
    @lru_cache(maxsize=1000)
    def _extract_claims(self, text: str) -> List[str]:
        """Extract factual claims using POS patterns"""
        try:
            grammar = r"""
                CLAIM: {<VBZ|VBD|VBP> <DT>? <JJ>* <NN.*>+ <VB.*>* <RB.*>*}
                FACT: {<IN>? <JJ>* <NN.*>+ <VBZ|VBD> <JJ>* <NN.*>+}
            """
            chunker = nltk.RegexpParser(grammar)
            tagged = nltk.pos_tag(word_tokenize(text))
            tree = chunker.parse(tagged)
            
            return [
                ' '.join(word for word, tag in subtree.leaves())
                for subtree in tree.subtrees()
                if subtree.label() in ['CLAIM', 'FACT']
            ]
        except Exception as e:
            logger.warning(f"Claim extraction failed: {e}")
            return []
    
    def _calculate_clarity(self, text: str) -> float:
        """Calculate clarity score based on readability"""
        sentences = sent_tokenize(text)
        if not sentences:
            return 0.0
            
        words = word_tokenize(text)
        avg_sent_len = len(words) / len(sentences)
        sent_len_score = 1 - (abs(avg_sent_len - 20) / 20)
        
        long_words = sum(1 for word in words if len(word) > 6)
        word_score = 1 - (long_words / len(words))
        
        return (0.6 * sent_len_score) + (0.4 * word_score)
    