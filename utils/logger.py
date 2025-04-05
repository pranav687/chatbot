import logging
import os
from utils.config import LOG_LEVEL, LOG_FILE

# Ensure the log directory exists
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Configure the logger
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def get_logger(name):
    """Get a logger instance with the given name"""
    return logging.getLogger(name) 