from typing import List
import re
import os

import ollama
from utils import config
from langchain.schema import Document
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.llms import Ollama

from utils.config import AI_MODEL, MEMORY_TYPE, VECTOR_DB_PATH, AI_TOPICS
from agents.base_agent import BaseAgent


class AIAgent(BaseAgent):
    """Agent for handling AI-related questions"""
    
    def __init__(self):
        """Initialize the AI agent."""
        system_prompt = """
        You are an AI expert assistant specializing in artificial intelligence, machine learning,
        neural networks, deep learning, and related technologies. Provide accurate, technically
        sound explanations and insights about AI concepts, methodologies, applications, and
        recent developments. When appropriate, include examples or analogies to make complex
        concepts more accessible.
        """
        
        super().__init__(
            model_name=AI_MODEL,
            system_prompt=system_prompt
        )
        
        # Initialize vector store
        self.vector_store = Chroma(
            collection_name="ai_knowledge",
            embedding_function=self.embeddings,
            persist_directory=config.VECTOR_DB_PATH + "/ai_agent"
        )
        
        # Initialize Wikipedia retriever for AI topics
        self.wiki_retriever = WikipediaRetriever()
        
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(model=AI_MODEL)
        
        # Initialize vector store
        self.vector_store_path = f"{VECTOR_DB_PATH}/ai_agent"
        
        # Create directory if it doesn't exist
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
                    self._initialize_knowledge_base()
            else:  # chroma
                # Create a new Chroma instance with proper configuration
                self.vector_store = Chroma(
                    persist_directory=self.vector_store_path,
                    embedding_function=self.embeddings,
                    collection_name="ai_knowledge"
                )
                
                # If no documents are in the vector store, add the AI topics
                if len(self.vector_store.get()["ids"]) == 0:
                    self._initialize_knowledge_base()
        except Exception as e:
            print(f"Error initializing vector store: {str(e)}")
            # Create a new vector store with default settings
            self.vector_store = Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=self.embeddings,
                collection_name="ai_knowledge"
            )
            self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Initialize the knowledge base with AI-related information"""
        # For each AI topic, get Wikipedia information and add to vector store
        documents = []
        
        self.logger.info("Initializing AI knowledge base with Wikipedia topics")
        
        # Add some basic information about each AI topic
        for topic in AI_TOPICS:
            try:
                self.logger.debug(f"Retrieving Wikipedia information for topic: {topic}")
                wiki_docs = self.wiki_retriever.get_relevant_documents(topic)
                
                if wiki_docs:
                    for doc in wiki_docs:
                        doc.metadata["topic"] = topic
                    documents.extend(wiki_docs)
                    self.logger.debug(f"Added {len(wiki_docs)} documents for topic: {topic}")
            except Exception as e:
                self.logger.error(f"Error retrieving Wikipedia info for {topic}: {e}")
                
                # Add a fallback document if Wikipedia retrieval fails
                fallback_content = f"Information about {topic} in artificial intelligence."
                documents.append(Document(
                    page_content=fallback_content,
                    metadata={"topic": topic, "source": "fallback"}
                ))
                self.logger.debug(f"Added fallback document for topic: {topic}")
        
        # Add documents to vector store
        if documents:
            self.logger.info(f"Adding {len(documents)} documents to vector store")
            if MEMORY_TYPE == "faiss":
                self.vector_store.add_documents(documents)
            else:  # chroma
                self.vector_store.add_documents(documents)
            self.logger.info("Knowledge base initialized successfully")
        else:
            self.logger.warning("No documents were added to the knowledge base")
    
    def retrieve_relevant_context(self, query: str) -> List[Document]:
        """Retrieve relevant context about AI topics from the knowledge base"""
        # Use the vector store to retrieve relevant documents
        self.logger.debug(f"Retrieving AI context for query: {query}")
        if self.vector_store:
            docs = self.vector_store.similarity_search(query, k=5)
            self.logger.debug(f"Retrieved {len(docs)} documents from vector store")
            return docs
        
        self.logger.warning("No vector store available for retrieval")
        return []
    
    # Method replaced by LLM-based classification in the IntentClassifier
    def can_handle_query(self, query: str) -> float:
        """For compatibility with old code - returns default confidence"""
        return 0.5
    
    def _generate_concise_response(self, query: str, context: str) -> str:
        prompt = f"""
        You are an AI expert assistant. Provide a concise and clear response.

        Context: {context}

        Question: {query}

        Short Answer:"""

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            options={"temperature": 0.3}
        )
        return response['message']['content'].strip()

    def _generate_detailed_response(self, query: str, context: str) -> str:
        prompt = f"""
        You are an AI expert assistant. Provide a detailed explanation, including relevant examples.

        Context: {context}

        Question: {query}

        Detailed Explanation:"""

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            options={"temperature": 0.7}
        )
        return response['message']['content'].strip()

    def _generate_clarifying_question(self, query: str, context: str) -> str:
        prompt = f"""
        You're uncertain about the user's AI-related question. Politely request more information.

        Context: {context}

        Question: {query}

        Clarifying Question:"""

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            options={"temperature": 0.5}
        )
        return response['message']['content'].strip()
