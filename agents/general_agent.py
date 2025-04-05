from typing import List
import re
import os
from utils import config
from langchain.schema import Document
from langchain_community.vectorstores import FAISS, Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.llms import Ollama
from langchain_community.chat_message_histories import ChatMessageHistory

from utils.config import GENERAL_MODEL, MEMORY_TYPE, VECTOR_DB_PATH
from agents.base_agent import BaseAgent


class GeneralAgent(BaseAgent):
    """Agent for handling general questions"""
    
    def __init__(self):
        """Initialize the general agent."""
        system_prompt = """
        You are a helpful assistant that answers general questions on various topics.
        Provide accurate, concise, and helpful responses based on your knowledge and
        the context provided. If you don't know the answer, admit it and suggest ways
        the user could find more information.
        """
        
        super().__init__(
            model_name=GENERAL_MODEL,
            system_prompt=system_prompt
        )
        
        # Initialize Wikipedia retriever for external knowledge
        self.wiki_retriever = WikipediaRetriever()
        
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(model=GENERAL_MODEL)
        
        # Initialize vector store
        self.vector_store = Chroma(
            collection_name="general_knowledge",
            embedding_function=self.embeddings,
            persist_directory=config.VECTOR_DB_PATH + "/general_agent"
        )
        
        # Create directory if it doesn't exist
        self.vector_store_path = f"{VECTOR_DB_PATH}/general_agent"
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        try:
            if MEMORY_TYPE == "faiss":
                if os.path.exists(self.vector_store_path) and os.listdir(self.vector_store_path):
                    self.vector_store = FAISS.load_local(
                        self.vector_store_path, 
                        self.embeddings
                    )
                else:
                    self.vector_store = FAISS.from_texts(
                        ["Initial document"], 
                        self.embeddings
                    )
            else:  # chroma
                # Create a new Chroma instance with proper configuration
                self.vector_store = Chroma(
                    persist_directory=self.vector_store_path,
                    embedding_function=self.embeddings,
                    collection_name="general_knowledge"
                )
        except Exception as e:
            print(f"Error initializing vector store: {str(e)}")
            # Create a new vector store with default settings
            self.vector_store = Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=self.embeddings,
                collection_name="general_knowledge"
            )
            
    def retrieve_relevant_context(self, query: str) -> List[Document]:
        """Retrieve relevant general information from multiple sources"""
        wiki_docs = []
        try:
            # Get information from Wikipedia
            wiki_docs = self.wiki_retriever.get_relevant_documents(query)
        except Exception as e:
            print(f"Error retrieving from Wikipedia: {e}")
        
        # If we have a vector store, search it as well
        local_docs = []
        if self.vector_store:
            try:
                local_docs = self.vector_store.similarity_search(query, k=3)
            except Exception as e:
                print(f"Error retrieving from vector store: {e}")
                
        # Combine and return results
        return wiki_docs + local_docs
    
    # Method replaced by LLM-based classification in the IntentClassifier
    def can_handle_query(self, query: str) -> float:
        """For compatibility with old code - returns default confidence"""
        return 0.5