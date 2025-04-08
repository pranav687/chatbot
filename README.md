# Multi-Agent Chatbot System: Execution Guide

This guide provides detailed instructions on how to set up and run the Multi-Agent Chatbot System.

Demo video is hosted here:http://drive.google.com/file/d/1f959k6x77eaAvGJKjbvq_PtbSdGiZ7Fs/view

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.8 or later
- [Ollama](https://ollama.ai/) for local LLM support
- Git (for cloning the repository)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/pranav687/chatbot.git
cd chatbot
```

### 2. Create and Activate a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root directory with the following variables:

```
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Ollama Model Configuration
GENERAL_MODEL=llama2
ADMISSION_MODEL=llama2
AI_MODEL=llama2
INTENT_MODEL=llama2

# Vector Store Configuration
VECTOR_STORE_PATH=./vector_db
CHROMA_PERSIST_DIRECTORY=./vector_db

# Memory Configuration
MEMORY_DIR=./conversation_memory

# Knowledge Base Configuration
ADMISSION_KB_PATH=./knowledge/admission_kb.txt
AI_KB_PATH=./knowledge/ai_kb.txt

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=./logs/chatbot.log

# Evaluation Configuration (for grad students)
EVALUATION_ENABLED=true
EVALUATION_METRICS=coherence,relevance,helpfulness

# Performance Configuration
MAX_TOKENS=4096
TEMPERATURE=0.7
TOP_P=0.95
TOP_K=40
REPEAT_PENALTY=1.1
```

Adjust these values as needed for your environment.

## Running the System

### 1. Start the Ollama Service

Ensure the Ollama service is running before starting the chatbot:

```bash
# On macOS/Linux this typically starts automatically after installation
# If needed, restart with:
ollama serve
```

### 2. Start the Chatbot Server

```bash
# From the project root directory, with virtual environment activated
python run.py
```

The server will start at http://localhost:8000 (or the port specified in your .env file).

### 3. Access the Web Interface

Open your browser and navigate to:

```
http://localhost:8000
```

## Testing the Chatbot

To test the chatbot's functionality, try the following types of questions:

1. **General knowledge questions** - e.g., "What is deep learning?"
2. **Concordia CS admission questions** - e.g., "What are the admission requirements for Concordia's CS program?"
3. **AI-specific questions** - e.g., "How do neural networks work?"
4. **Context-aware follow-up questions** - e.g., "Tell me more about it" (after asking about a specific topic)

## Troubleshooting

1. **Ollama connection issues**:

   - Ensure Ollama is running with `ps aux | grep ollama`
   - Check the OLLAMA_BASE_URL in your .env file
   - Verify network connectivity to the Ollama service
2. **Missing dependencies**:

   - Ensure all requirements are installed with `pip install -r requirements.txt`
3. **Port conflicts**:

   - If port 8000 is already in use, change the PORT value in your .env file

## API Usage

For developers who want to integrate with the chatbot API:

### Chat Endpoint

```
POST http://localhost:8000/api/chat
```

Request body:

```json
{
  "query": "Your question here",
  "conversation_id": "optional-existing-conversation-id for context awarenerss"
}
```

Response:

```json
{
  "answer": "The chatbot's response",
  "conversation_id": "conversation-id",
  "agent_type": "Name of the agent that handled the query",
  "confidence": <0-1>
}
```

## Monitoring and Logs

Logs are printed to the console by default. The verbosity is controlled by the LOG_LEVEL environment variable. For debugging, set LOG_LEVEL=DEBUG in your .env file.
