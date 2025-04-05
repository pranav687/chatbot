import os
from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import OllamaEmbeddings

from utils.config import VECTOR_DB_PATH, MEMORY_TYPE


class VectorStore:
    """Wrapper for vector database operations"""
    
    def __init__(self, collection_name: str, model_name: str = "llama2"):
        """
        Initialize vector store
        
        Args:
            collection_name: Name of the collection/index
            model_name: Name of the Ollama model to use for embeddings
        """
        self.collection_name = collection_name
        self.embeddings = OllamaEmbeddings(model=model_name)
        self.vector_store_path = f"{VECTOR_DB_PATH}/{collection_name}"
        
        # Create directory if it doesn't exist
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        # Initialize vector store
        self.vector_store = self._initialize_store()
    
    def _initialize_store(self):
        """Initialize or load the vector store"""
        try:
            if MEMORY_TYPE == "faiss":
                return FAISS.load_local(
                    self.vector_store_path, 
                    self.embeddings
                )
            else:  # chroma
                return Chroma(
                    persist_directory=self.vector_store_path,
                    embedding_function=self.embeddings
                )
        except Exception as e:
            print(f"Error loading vector store: {e}")
            
            # Create a new store
            if MEMORY_TYPE == "faiss":
                store = FAISS.from_texts(
                    ["Initialization document for " + self.collection_name],
                    self.embeddings
                )
                store.save_local(self.vector_store_path)
                return store
            else:  # chroma
                return Chroma(
                    persist_directory=self.vector_store_path,
                    embedding_function=self.embeddings
                )
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Add texts to the vector store
        
        Args:
            texts: List of text strings
            metadatas: Optional list of metadata dicts
            
        Returns:
            List of document IDs
        """
        if MEMORY_TYPE == "faiss":
            return self.vector_store.add_texts(texts, metadatas)
        else:  # chroma
            ids = [f"{self.collection_name}_{i}" for i in range(len(texts))]
            self.vector_store.add_texts(texts, metadatas, ids)
            return ids
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of document IDs
        """
        if MEMORY_TYPE == "faiss":
            return self.vector_store.add_documents(documents)
        else:  # chroma
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            ids = [f"{self.collection_name}_{i}" for i in range(len(texts))]
            self.vector_store.add_texts(texts, metadatas, ids)
            return ids
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Search for similar documents
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of Document objects
        """
        return self.vector_store.similarity_search(query, k=k)
    
    def persist(self):
        """Save the vector store to disk"""
        if MEMORY_TYPE == "faiss":
            self.vector_store.save_local(self.vector_store_path)
        else:  # chroma
            if hasattr(self.vector_store, "persist"):
                self.vector_store.persist()
    
    def clear(self):
        """Clear the vector store"""
        if MEMORY_TYPE == "chroma":
            self.vector_store.delete_collection()
            self.vector_store = self._initialize_store()
        else:  # faiss
            # For FAISS, we need to recreate the store
            if os.path.exists(self.vector_store_path):
                import shutil
                shutil.rmtree(self.vector_store_path)
            
            self.vector_store = self._initialize_store()