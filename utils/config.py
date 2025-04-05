import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GENERAL_MODEL = os.getenv("GENERAL_MODEL", "llama2")
ADMISSION_MODEL = os.getenv("ADMISSION_MODEL", "llama2")
AI_MODEL = os.getenv("AI_MODEL", "llama2")
INTENT_MODEL = os.getenv("INTENT_MODEL", "llama2")  # Model for intent classification

# Ollama settings dictionary
OLLAMA_SETTINGS = {
    "GENERAL_MODEL": GENERAL_MODEL,
    "ADMISSION_MODEL": ADMISSION_MODEL,
    "AI_MODEL": AI_MODEL,
    "INTENT_MODEL": INTENT_MODEL,
    "OLLAMA_BASE_URL": OLLAMA_BASE_URL
}

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8001"))

# Memory settings
MEMORY_TYPE = os.getenv("MEMORY_TYPE", "chroma")  # 'faiss' or 'chroma'
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_db")

# External API settings
WIKIPEDIA_USER_AGENT = os.getenv("WIKIPEDIA_USER_AGENT", "Concordia-Chatbot/1.0")

# Concordia specific data
CONCORDIA_CS_PROGRAM_INFO = {
    "name": "Computer Science",
    "degrees": ["Bachelor of Computer Science (BCompSc)", "Master of Computer Science (MCompSc)", "PhD in Computer Science"],
    "departments": ["Department of Computer Science and Software Engineering"],
    "faculty": "Faculty of Engineering and Computer Science",
    "admission_requirements": {
        "undergraduate": {
            "high_school": "High school diploma with advanced math and science courses",
            "cegep": "DEC with advanced math and science courses",
            "minimum_gpa": 3.0
        },
        "graduate": {
            "masters": "Bachelor's degree in Computer Science or related field with minimum 3.0 GPA",
            "phd": "Master's degree in Computer Science or related field with research experience"
        }
    },
    "application_deadlines": {
        "fall": "March 1",
        "winter": "November 1",
        "summer": "Not all programs accept summer admissions"
    },
    "tuition_fees": {
        "quebec_residents": "Approximately $4,000 per year",
        "canadian_non_quebec": "Approximately $9,000 per year",
        "international": "Approximately $27,000 per year"
    },
    "program_website": "https://www.concordia.ca/academics/undergraduate/computer-science.html"
}

# AI-specific topics for demonstration
AI_TOPICS = [
    "machine learning",
    "deep learning",
    "neural networks",
    "natural language processing",
    "computer vision",
    "reinforcement learning",
    "generative ai",
    "large language models",
    "transformers",
    "artificial intelligence ethics",
    "ai history",
    "ai applications",
    "rag",
    "llm",
    "llm history",
    "llm applications",
    "llm ethics",
    "llm history",
    "llm applications",
    "llm ethics",
    "llm history",
    "llm applications",
    "llm ethics",
    "llm history",
    "llm applications",
    "llm ethics",
]

# Vector Store Configuration
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./vector_db")
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./vector_db")

# Memory Configuration
MEMORY_DIR = os.getenv("MEMORY_DIR", "./conversation_memory")

# Knowledge Base Configuration
ADMISSION_KB_PATH = os.getenv("ADMISSION_KB_PATH", "./knowledge/admission_kb.txt")
AI_KB_PATH = os.getenv("AI_KB_PATH", "./knowledge/ai_kb.txt")
TEST_KB_PATH = os.getenv("TEST_KB_PATH", "./knowledge/test_kb.txt")

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "./logs/chatbot.log")

# Evaluation Configuration (for grad students)
EVALUATION_ENABLED = os.getenv("EVALUATION_ENABLED", "true").lower() == "true"
EVALUATION_METRICS = os.getenv("EVALUATION_METRICS", "coherence,relevance,helpfulness").split(",")

# Security Configuration
API_KEY_REQUIRED = os.getenv("API_KEY_REQUIRED", "false").lower() == "true"
API_KEY = os.getenv("API_KEY", "your_api_key_here")

# Performance Configuration
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.95"))
TOP_K = int(os.getenv("TOP_K", "40"))
REPEAT_PENALTY = float(os.getenv("REPEAT_PENALTY", "1.1"))