from typing import List, Dict, Any
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from agents.base_agent import BaseAgent
from utils.config import TEST_KB_PATH, GENERAL_MODEL


class TestAgent(BaseAgent):
    """Agent for testing vector database functionality"""
    
    def __init__(self):
        """Initialize the test agent"""
        super().__init__(
            model_name=GENERAL_MODEL,
            system_prompt="You are a helpful assistant for testing the vector database functionality. Use the provided context to answer questions.",
            temperature=0.7
        )
        
        # Initialize vector store with test knowledge base
        self.vector_store = self._initialize_vector_store(TEST_KB_PATH)
    
    def can_handle_query(self, query: str) -> float:
        """For compatibility with old code - returns default confidence"""
        return 0.5
    
    def retrieve_relevant_context(self, query: str, k: int = 3) -> List[Document]:
        """Retrieve relevant context from the vector store"""
        if not self.vector_store:
            print(f"[DEBUG] No vector store available for {self.__class__.__name__}")
            return []
        
        # Search for relevant documents
        results = self.vector_store.similarity_search(query, k=k)
        
        # Debug logging
        print(f"[DEBUG] Retrieved {len(results)} documents from vector store for query: '{query}'")
        for i, doc in enumerate(results):
            print(f"[DEBUG] Document {i+1}: {doc.page_content[:100]}...")
        
        return results 