from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from utils import config
import ollama
import json
import os
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import OllamaLLM
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_chroma import Chroma

from utils.config import OLLAMA_BASE_URL
from utils.logger import get_logger


class BaseAgent(ABC):
    """Base class for all chatbot agents"""
    
    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        memory_key: str = "chat_history",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.memory_key = memory_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = get_logger(self.__class__.__name__)
        
        self.logger.info(f"Initializing {self.__class__.__name__} with model: {model_name}")
        
        # Initialize the Ollama model
        self.llm = OllamaLLM(
            model=model_name,
            base_url=OLLAMA_BASE_URL,
            temperature=temperature
        )
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key=memory_key,
            return_messages=True,
            output_key="output"
        )
        
        # Default response template
        self.qa_template = PromptTemplate(
            input_variables=["context", "question", memory_key],
            template=f"""
            System: {system_prompt}
            
            Context: {{context}}
            
            Chat History: {{{memory_key}}}
            
            User: {{question}}
            
            Assistant:"""
        )

        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(
            model=config.OLLAMA_SETTINGS["GENERAL_MODEL"],
            base_url=config.OLLAMA_SETTINGS["OLLAMA_BASE_URL"]
        )
        
        self.logger.info(f"{self.__class__.__name__} initialized successfully")
    
    @abstractmethod
    def retrieve_relevant_context(self, query: str) -> List[Document]:
        """Retrieve relevant context for the query from the knowledge base"""
        pass
    
    def can_handle_query(self, query: str) -> float:
        """Default confidence scorer, overridden by specialized agents if needed"""
        return 0.5
    
    def process_query(self, query: str, conversation_id: str = None) -> Dict[str, Any]:
        """Process the query and return a response"""
        self.logger.info(f"Processing query: '{query}'")
        
        # Retrieve relevant context
        self.logger.debug("Retrieving relevant context")
        relevant_docs = self.retrieve_relevant_context(query)
        context = "\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else ""
        
        # Get conversation history
        history = self.memory.chat_memory.messages if hasattr(self.memory, 'chat_memory') else []
        history_text = ""
        
        # Format conversation history
        if history:
            self.logger.debug(f"Using conversation history with {len(history)} messages")
            for i in range(0, len(history), 2):
                if i+1 < len(history):
                    history_text += f"User: {history[i].content}\nAssistant: {history[i+1].content}\n\n"
        
        # Create a more context-aware prompt
        prompt = f"""Context: {context}

Previous conversation:
{history_text}

User query: {query}

IMPORTANT: If the user's query contains pronouns like "it", "there", "that", etc., these pronouns refer to topics discussed in the previous conversation. Make sure to interpret these pronouns correctly based on the conversation history.

Please provide a helpful and accurate response to the user's query. If the query refers to previous information in the conversation, make sure to maintain that context."""
        
        self.logger.debug(f"Sending prompt to Ollama model: {self.model_name}")
        
        # Generate response using direct Ollama API for more control
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": self.temperature,
                    "top_p": 0.95,
                    "top_k": 40,
                    "num_ctx": 4096,
                    "repeat_penalty": 1.1
                }
            )
            
            answer = response['message']['content']
            self.logger.debug(f"Generated response: {answer[:100]}...")
            
            # Update memory
            self.memory.save_context({"input": query}, {"output": answer})
            self.logger.debug("Updated conversation memory")
            
            return {
                "answer": answer,
                "source": "agent",
                "agent_type": self.__class__.__name__,
                "confidence": self.can_handle_query(query),
                "context_used": bool(relevant_docs)
            }
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            # Return a fallback response
            return {
                "answer": "I'm sorry, I encountered an error processing your request. Please try again.",
                "source": "agent",
                "agent_type": self.__class__.__name__,
                "confidence": 0.0,
                "context_used": False,
                "error": str(e)
            }
    
    def get_memory(self) -> List[Dict[str, str]]:
        """Return the conversation history"""
        self.logger.debug("Retrieving conversation memory")
        return self.memory.chat_memory.messages
    
    def clear_memory(self):
        """Clear the conversation history"""
        self.logger.info("Clearing conversation memory")
        self.memory.clear()

    def _initialize_vector_store(self, knowledge_base_path: str):
        """Initialize the vector store with the given knowledge base"""
        self.logger.info(f"Initializing vector store with knowledge base: {knowledge_base_path}")
        
        if not os.path.exists(knowledge_base_path):
            self.logger.warning(f"Knowledge base file not found: {knowledge_base_path}")
            return None
            
        try:
            # Read the knowledge base file
            with open(knowledge_base_path, 'r') as f:
                content = f.read()
            
            # Create documents from the content
            documents = [Document(page_content=content)]
            
            # Create and return the vector store
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=os.path.join("vector_db", self.__class__.__name__.lower())
            )
            self.logger.info(f"Vector store initialized successfully with {len(documents)} documents")
            return vector_store
        except Exception as e:
            self.logger.error(f"Failed to initialize vector store: {str(e)}")
            return None

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get default parameters for the LLM."""
        return {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "num_ctx": 4096,
            "repeat_penalty": 1.1
        }