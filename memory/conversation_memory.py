from typing import Dict, List, Any, Optional
import json
import os
from datetime import datetime


class ConversationMemory:
    """Stores and manages conversation history"""
    
    def __init__(self, memory_dir: str = "./conversation_memory"):
        """
        Initialize conversation memory
        
        Args:
            memory_dir: Directory to store conversation history
        """
        self.memory_dir = memory_dir
        
        # Create directory if it doesn't exist
        os.makedirs(memory_dir, exist_ok=True)
        
        # In-memory store for active conversations
        self.active_conversations = {}
    
    def add_message(self, conversation_id: str, role: str, content: str, 
                   metadata: Optional[Dict[str, Any]] = None):
        """
        Add a message to conversation history
        
        Args:
            conversation_id: Unique ID for the conversation
            role: 'user' or 'assistant'
            content: Message content
            metadata: Optional metadata about the message
        """
        # Initialize conversation if not exists
        if conversation_id not in self.active_conversations:
            self.active_conversations[conversation_id] = []
        
        # Create message object
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Add metadata if provided
        if metadata:
            message["metadata"] = metadata
        
        # Add to active conversations
        self.active_conversations[conversation_id].append(message)
        
        # Persist to disk
        self._save_conversation(conversation_id)
    
    def _save_conversation(self, conversation_id: str):
        """Save conversation to disk"""
        file_path = os.path.join(self.memory_dir, f"{conversation_id}.json")
        
        with open(file_path, 'w') as f:
            json.dump(self.active_conversations[conversation_id], f, indent=2)
    
    def load_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Load conversation from disk
        
        Args:
            conversation_id: Unique ID for the conversation
            
        Returns:
            List of message objects
        """
        file_path = os.path.join(self.memory_dir, f"{conversation_id}.json")
        
        # Check if already in memory
        if conversation_id in self.active_conversations:
            return self.active_conversations[conversation_id]
        
        # Check if file exists
        if not os.path.exists(file_path):
            return []
        
        # Load from disk
        try:
            with open(file_path, 'r') as f:
                conversation = json.load(f)
                self.active_conversations[conversation_id] = conversation
                return conversation
        except Exception as e:
            print(f"Error loading conversation {conversation_id}: {e}")
            return []
    
    def get_conversation_history(self, conversation_id: str, 
                               max_messages: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get conversation history
        
        Args:
            conversation_id: Unique ID for the conversation
            max_messages: Optional maximum number of messages to return
            
        Returns:
            List of message objects
        """
        # Load conversation if not in memory
        if conversation_id not in self.active_conversations:
            self.load_conversation(conversation_id)
        
        # Get messages
        messages = self.active_conversations.get(conversation_id, [])
        
        # Limit number of messages if specified
        if max_messages and len(messages) > max_messages:
            return messages[-max_messages:]
        
        return messages
    
    def clear_conversation(self, conversation_id: str):
        """
        Clear conversation history
        
        Args:
            conversation_id: Unique ID for the conversation
        """
        # Remove from memory
        if conversation_id in self.active_conversations:
            del self.active_conversations[conversation_id]
        
        # Remove from disk
        file_path = os.path.join(self.memory_dir, f"{conversation_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
    
    def list_conversations(self) -> List[str]:
        """
        List all stored conversations
        
        Returns:
            List of conversation IDs
        """
        conversations = []
        
        # List files in memory directory
        for filename in os.listdir(self.memory_dir):
            if filename.endswith('.json'):
                conversations.append(filename[:-5])  # Remove .json extension
        
        return conversations