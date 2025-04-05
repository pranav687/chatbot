#!/usr/bin/env python
"""
Run script for the Concordia Multi-Agent Chatbot
This script loads environment variables from .env and starts the FastAPI server
"""

import os
import sys
import uvicorn
from dotenv import load_dotenv
from utils.logger import get_logger

# Load environment variables from .env file
load_dotenv()

# Setup logger
logger = get_logger("run")

# Get API configuration from environment variables
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8001"))

def main():
    """Main function to run the application"""
    logger.info(f"Starting Concordia Multi-Agent Chatbot on {API_HOST}:{API_PORT}")
    logger.info("Press Ctrl+C to stop the server")
    
    # Start the FastAPI server
    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nServer stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        sys.exit(1) 