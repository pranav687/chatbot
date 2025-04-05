from typing import Dict, Any, List
from uuid import uuid4

from orchestrator.intent_classifier import IntentClassifier
from agents.base_agent import BaseAgent
from utils.logger import get_logger


class Router:
    """Routes user queries to appropriate agents and manages conversations"""
    
    def __init__(self, agents: List[BaseAgent]):
        """
        Initialize router with a list of agent instances
        
        Args:
            agents: List of agent instances
        """
        self.agents = agents
        self.intent_classifier = IntentClassifier(agents)
        self.conversations = {}  # Store conversation state by ID
        self.confidence_threshold = 0.5  # Minimum confidence to use a specialized agent
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info(f"Router initialized with {len(agents)} agents and confidence threshold {self.confidence_threshold}")
    
    def process_query(self, query: str, conversation_id: str = None) -> Dict[str, Any]:
        """
        Process a user query and return a response
        
        Args:
            query: User query string
            conversation_id: Optional conversation ID for continuing a conversation
            
        Returns:
            Dict containing the response and metadata
        """
        # Create a new conversation ID if none provided
        if not conversation_id:
            conversation_id = str(uuid4())
            self.logger.info(f"Created new conversation with ID: {conversation_id}")
            
        # Get conversation state or create new one
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = {
                "history": [],
                "last_agent": None
            }
            self.logger.info(f"Initialized new conversation state for ID: {conversation_id}")
        
        self.logger.debug(f"Processing query for conversation {conversation_id}: {query}")
        
        # Get conversation history for context
        history = self.conversations[conversation_id]["history"]
        
        # Check if this is a follow-up question with pronouns
        is_follow_up = False
        pronouns = ["it", "they", "them", "those", "these", "there", "here", "that", "this"]
        for pronoun in pronouns:
            if pronoun in query.lower():
                is_follow_up = True
                break
        
        # If it's a follow-up question and we have a last agent, use it
        if is_follow_up and self.conversations[conversation_id]["last_agent"]:
            best_agent = self.conversations[conversation_id]["last_agent"]
            confidence = 1.0  # High confidence to use the same agent
            self.logger.info(f"Using previous agent for follow-up question: {best_agent.__class__.__name__}")
        else:
            # Classify the intent to select the appropriate agent
            self.logger.debug("Classifying intent to select appropriate agent")
            best_agent, confidence = self.intent_classifier.classify_intent(query, history)
            
            # If confidence is below threshold, use the last agent if available
            if confidence < self.confidence_threshold and self.conversations[conversation_id]["last_agent"]:
                self.logger.info(f"Confidence ({confidence}) below threshold ({self.confidence_threshold}), using previous agent")
                best_agent = self.conversations[conversation_id]["last_agent"]
                confidence = self.confidence_threshold  # Set to threshold to indicate continued conversation
        
        # Process the query with the selected agent
        self.logger.info(f"Processing query with agent: {best_agent.__class__.__name__}, confidence: {confidence}")
        response = best_agent.process_query(query, conversation_id)
        
        # Store the conversation state
        self.conversations[conversation_id]["history"].append({
            "query": query,
            "response": response["answer"],
            "agent_type": response["agent_type"]
        })
        self.conversations[conversation_id]["last_agent"] = best_agent
        self.logger.debug(f"Updated conversation history for {conversation_id}, now has {len(history) + 1} messages")
        
        # Add conversation_id to response
        response["conversation_id"] = conversation_id
        
        return response
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """
        Get the conversation history for a specific conversation
        
        Args:
            conversation_id: The ID of the conversation
            
        Returns:
            List of query/response pairs
        """
        self.logger.debug(f"Retrieving conversation history for ID: {conversation_id}")
        if conversation_id in self.conversations:
            return self.conversations[conversation_id]["history"]
        else:
            self.logger.warning(f"No conversation found with ID: {conversation_id}")
            return []
    
    def clear_conversation(self, conversation_id: str) -> bool:
        """
        Clear a conversation from memory
        
        Args:
            conversation_id: The ID of the conversation to clear
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Attempting to clear conversation with ID: {conversation_id}")
        if conversation_id in self.conversations:
            # Clear the agent memories
            if self.conversations[conversation_id]["last_agent"]:
                agent_type = self.conversations[conversation_id]["last_agent"].__class__.__name__
                self.logger.info(f"Clearing memory for agent: {agent_type}")
                self.conversations[conversation_id]["last_agent"].clear_memory()
                
            # Remove the conversation
            del self.conversations[conversation_id]
            self.logger.info(f"Successfully cleared conversation: {conversation_id}")
            return True
        else:
            self.logger.warning(f"Failed to clear conversation: {conversation_id} - not found")
            return False