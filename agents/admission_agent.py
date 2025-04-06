from typing import Dict, List
import re
import os
from langchain.schema import Document
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
import ollama

from utils.config import ADMISSION_MODEL, MEMORY_TYPE, VECTOR_DB_PATH, CONCORDIA_CS_PROGRAM_INFO
from agents.base_agent import BaseAgent


class AdmissionAgent(BaseAgent):
    """Agent for handling questions about Concordia CS program admissions"""
    
    def __init__(self):
        """Initialize the admission agent."""
        system_prompt = """
        You are a specialized assistant for Concordia University's Computer Science program admissions.
        Your role is to provide accurate information about admission requirements, application processes,
        deadlines, program details, and other relevant information for prospective students.
        Be helpful, accurate, and supportive to students considering Concordia's CS programs.
        """
        
        super().__init__(
            model_name=ADMISSION_MODEL,
            system_prompt=system_prompt
        )
        
        # Initialize vector store
        self.vector_store = Chroma(
            collection_name="admission_knowledge",
            embedding_function=self.embeddings,
            persist_directory=VECTOR_DB_PATH + "/admission_agent"
        )
        
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(model=ADMISSION_MODEL)
        
        # Initialize vector store
        self.vector_store_path = f"{VECTOR_DB_PATH}/admission_agent"
        
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
                # If no documents are in the vector store, add the program info
                if len(self.vector_store.get()["ids"]) == 0:
                    self._initialize_knowledge_base()
                    
        except Exception as e:
            print(f"Error initializing vector store: {str(e)}")
            # Create a new vector store with default settings
            self.vector_store = Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=self.embeddings,
                collection_name="admission_knowledge"
            )
            self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Initialize the knowledge base with Concordia CS program information"""
        # Convert the program info dictionary to text chunks
        documents = []
        
        # Basic program information
        basic_info = f"""
        Program Name: {CONCORDIA_CS_PROGRAM_INFO['name']}
        Faculty: {CONCORDIA_CS_PROGRAM_INFO['faculty']}
        Department: {', '.join(CONCORDIA_CS_PROGRAM_INFO['departments'])}
        Degrees Offered: {', '.join(CONCORDIA_CS_PROGRAM_INFO['degrees'])}
        Program Website: {CONCORDIA_CS_PROGRAM_INFO['program_website']}
        """
        documents.append(Document(page_content=basic_info, metadata={"source": "program_info"}))
        
        # Admission requirements
        undergrad_req = f"""
        Undergraduate Admission Requirements for Computer Science at Concordia:
        - High School Requirements: {CONCORDIA_CS_PROGRAM_INFO['admission_requirements']['undergraduate']['high_school']}
        - CEGEP Requirements: {CONCORDIA_CS_PROGRAM_INFO['admission_requirements']['undergraduate']['cegep']}
        - Minimum GPA: {CONCORDIA_CS_PROGRAM_INFO['admission_requirements']['undergraduate']['minimum_gpa']}
        """
        documents.append(Document(page_content=undergrad_req, metadata={"source": "undergrad_requirements"}))
        
        grad_req = f"""
        Graduate Admission Requirements for Computer Science at Concordia:
        - Master's Program: {CONCORDIA_CS_PROGRAM_INFO['admission_requirements']['graduate']['masters']}
        - PhD Program: {CONCORDIA_CS_PROGRAM_INFO['admission_requirements']['graduate']['phd']}
        """
        documents.append(Document(page_content=grad_req, metadata={"source": "grad_requirements"}))
        
        # Application deadlines
        deadlines = f"""
        Application Deadlines for Concordia Computer Science Programs:
        - Fall Term: {CONCORDIA_CS_PROGRAM_INFO['application_deadlines']['fall']}
        - Winter Term: {CONCORDIA_CS_PROGRAM_INFO['application_deadlines']['winter']}
        - Summer Term: {CONCORDIA_CS_PROGRAM_INFO['application_deadlines']['summer']}
        """
        documents.append(Document(page_content=deadlines, metadata={"source": "deadlines"}))
        
        # Tuition fees
        tuition = f"""
        Tuition Fees for Concordia Computer Science Programs:
        - Quebec Residents: {CONCORDIA_CS_PROGRAM_INFO['tuition_fees']['quebec_residents']}
        - Canadian Non-Quebec Residents: {CONCORDIA_CS_PROGRAM_INFO['tuition_fees']['canadian_non_quebec']}
        - International Students: {CONCORDIA_CS_PROGRAM_INFO['tuition_fees']['international']}
        """
        documents.append(Document(page_content=tuition, metadata={"source": "tuition"}))
        
        # Add documents to vector store
        if MEMORY_TYPE == "faiss":
            self.vector_store.add_documents(documents)
        else:  # chroma
            self.vector_store.add_documents(documents)
    
    def retrieve_relevant_context(self, query: str) -> List[Document]:
        """Retrieve relevant context about admissions from the vector store"""
        if self.vector_store:
            try:
                return self.vector_store.similarity_search(query, k=3)
            except Exception as e:
                print(f"Error retrieving from vector store: {e}")
        
        return []
    
    def _generate_concise_response(self, query: str, context: str) -> str:
        prompt = f"""
        You are a Concordia University admission assistant. Provide a short, precise answer.

        Context: {context}

        Question: {query}

        Answer:"""

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
        You are a Concordia University admission assistant. Provide a comprehensive, detailed answer.

        Context: {context}

        Question: {query}

        Detailed Answer:"""

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            options={"temperature": 0.6}
        )
        return response['message']['content'].strip()

    def _generate_clarifying_question(self, query: str, context: str) -> str:
        prompt = f"""
        You are unsure about the user's admission query. Politely ask a clarifying question.

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
