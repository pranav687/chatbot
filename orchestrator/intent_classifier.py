from typing import Dict, List, Tuple
import ollama
import json
import re
from utils.config import OLLAMA_BASE_URL, OLLAMA_SETTINGS
from utils.logger import get_logger

class IntentClassifier:
    """Classifies user intent to route to appropriate agent using LLM"""
    
    def __init__(self, agents: List):
        """Initialize with a list of agent instances"""
        self.agents = agents
        self.model_name = OLLAMA_SETTINGS.get("INTENT_MODEL", "llama2")
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info(f"IntentClassifier initialized with model: {self.model_name}")
    
    def classify_intent(self, query: str, conversation_history: List[Dict] = None) -> Tuple[object, float]:
        """
        Use Ollama LLM to determine which agent should handle this query
        
        Args:
            query: The user's query string
            conversation_history: Optional list of previous messages in the conversation
            
        Returns:
            Tuple containing (best_agent, confidence_score)
        """
        self.logger.debug(f"Classifying intent for query: {query}")
        
        
        agent_descriptions = []
        for agent in self.agents:
            agent_type = agent.__class__.__name__
            # Strip "Agent" suffix if present
            if agent_type.endswith("Agent"):
                agent_type = agent_type[:-5]
            agent_descriptions.append(f"- {agent_type}: {self._get_agent_description(agent)}")
        
        agent_info = "\n".join(agent_descriptions)
        
        prompt = f"""
Based on the following user query, determine which agent would be most appropriate to handle it.

User query: "{query}"

Available agents:
{agent_info}

The response should be a JSON object with two fields:
1. "agent": The name of the agent that should handle this query
2. "confidence": A confidence score between 0.0 and 1.0

Only respond with the JSON object, nothing else.
        """
        
        self.logger.debug(f"Sending prompt to Ollama model: {self.model_name}")
        
        # Call Ollama API
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful intent classifier that determines which agent should handle a query."},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": 0.1,  # Keep temperature low for consistent results
                    "top_p": 0.95,
                    "num_ctx": 2048
                }
            )
            
            content = response['message']['content']
            self.logger.debug(f"Received response from Ollama: {content[:100]}...")
            
            # Parse the JSON response
            # First, find the JSON object in the response (it might include markdown code blocks)
            
            # Try to extract JSON from the response if it's wrapped in code blocks
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
            if json_match:
                json_str = json_match.group(1)
                self.logger.debug("Extracted JSON from code block")
            else:
                json_str = content
                
            try:
                result = json.loads(json_str)
                agent_name = result.get("agent", "").strip()
                confidence = result.get("confidence", 0.0)
                
                self.logger.info(f"LLM selected agent: {agent_name} with confidence: {confidence}")
                
                # Find the agent matching the name
                for agent in self.agents:
                    class_name = agent.__class__.__name__
                    # Check for exact match or if agent name is contained in class name
                    if class_name == f"{agent_name}Agent" or class_name == agent_name:
                        self.logger.info(f"Selected agent: {class_name}, Confidence: {confidence}")
                        return agent, confidence
                
                # Fall back to the first agent if no match found
                self.logger.warning(f"No matching agent found for {agent_name}, falling back to default")
                return self.agents[0], 0.5
                
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse JSON from LLM response: {content}")
                return self.agents[0], 0.5
                
        except Exception as e:
            self.logger.error(f"Error calling Ollama: {str(e)}")
            return self.agents[0], 0.5
    
    def _get_agent_description(self, agent) -> str:
        """Get a description of what the agent handles"""
        agent_type = agent.__class__.__name__
        
        # Predefined descriptions for known agent types
        descriptions = {
            "GeneralAgent": "Handles general queries about various topics, serving as a fallback for questions that don't fit other specialized categories.",
            "AdmissionAgent": "Specializes in university admissions, application processes, program information, tuition fees, and student enrollment.",
            "AIAgent": "Expert in artificial intelligence, machine learning, neural networks, deep learning, NLP, and related technologies.",
            "TestAgent": "Used for testing purposes only."
        }
        
        return descriptions.get(agent_type, "No description available")