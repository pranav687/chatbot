import os
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import uvicorn
import json

from utils.config import API_HOST, API_PORT, GENERAL_MODEL, ADMISSION_MODEL, AI_MODEL
from agents.general_agent import GeneralAgent
from agents.admission_agent import AdmissionAgent
from agents.ai_agent import AIAgent
from agents.test_agent import TestAgent
from orchestrator.router import Router
from memory.conversation_memory import ConversationMemory
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory  # Check migration guide for updated import
from evaluation.rl_agent import RLAgent, State, Action, Reward
from evaluation.metrics import ResponseEvaluator

# Initialize FastAPI
app = FastAPI(
    title="Concordia Multi-Agent Chatbot",
    description="A multi-agent chatbot system using Ollama for intelligent conversations",
    version="1.0.0"
)

# Initialize agents, router, memory, and evaluation components
general_agent = GeneralAgent()
admission_agent = AdmissionAgent()
ai_agent = AIAgent()
test_agent = TestAgent()

router = Router([general_agent, admission_agent, ai_agent, test_agent])
memory = ConversationMemory()
rl_agent = RLAgent()
response_evaluator = ResponseEvaluator()

# Pydantic models for request and response
class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    agent_type: str
    confidence: float

class ConversationHistoryResponse(BaseModel):
    history: List[Dict[str, Any]]

class ClearConversationRequest(BaseModel):
    conversation_id: str

class ListConversationsResponse(BaseModel):
    conversations: List[str]


@app.get("/")
async def root():
    """Home endpoint with basic info"""
    return {
        "message": "Concordia Multi-Agent Chatbot API",
        "version": "1.0.0",
        "endpoints": [
            "/chat", 
            "/conversations", 
            "/conversations/{conversation_id}", 
            "/conversations/{conversation_id}/clear"
        ]
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat request"""
    try:
        # Get conversation history if conversation_id is provided
        conversation_history = []
        if request.conversation_id:
            conversation_history = memory.get_conversation_history(request.conversation_id)
        
        # Process the query
        response = router.process_query(request.query, request.conversation_id)
        
        # Save to memory
        memory.add_message(
            conversation_id=response["conversation_id"],
            role="user",
            content=request.query
        )
        memory.add_message(
            conversation_id=response["conversation_id"],
            role="assistant",
            content=response["answer"],
            metadata={"agent_type": response["agent_type"]}
        )
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.get("/conversations", response_model=ListConversationsResponse)
async def list_conversations():
    """List all conversations"""
    try:
        conversations = memory.list_conversations()
        return {"conversations": conversations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing conversations: {str(e)}")

@app.get("/conversations/{conversation_id}", response_model=ConversationHistoryResponse)
async def get_conversation(conversation_id: str):
    """Get conversation history"""
    try:
        history = memory.get_conversation_history(conversation_id)
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting conversation: {str(e)}")

@app.post("/conversations/{conversation_id}/clear")
async def clear_conversation(conversation_id: str):
    """Clear a conversation"""
    try:
        router.clear_conversation(conversation_id)
        memory.clear_conversation(conversation_id)
        return {"message": f"Conversation {conversation_id} cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing conversation: {str(e)}")

@app.post("/evaluation/{conversation_id}")
async def evaluate_conversation(
    conversation_id: str,
    request: Dict[str, Any]
) -> Dict[str, Any]:
    """Evaluate a conversation using RL-based metrics (Grad students only)"""
    try:
        # Get conversation history
        history = memory.get_conversation_history(conversation_id)
        if not history:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Initialize evaluation results
        evaluation_results = {
            'conversation_id': conversation_id,
            'metrics': [],
            'overall_score': 0.0,
            'rl_metrics': {}
        }
        
        # Evaluate each response in the conversation
        for i in range(1, len(history), 2):  # Skip user messages
            if i >= len(history):
                break
                
            query = history[i-1]['content']
            response = history[i]['content']
            agent_type = history[i].get('agent_type', 'general')
            
            # Calculate metrics
            metrics = response_evaluator.evaluate_response(query, response)
            
            # Create state and action for RL
            state = State(
                query=query,
                context=[msg['content'] for msg in history[:i]],
                previous_responses=[msg['content'] for msg in history[:i] if msg.get('role') == 'assistant'],
                agent_type=agent_type
            )
            
            action = Action(
                response=response,
                confidence=history[i].get('confidence', 0.5),
                agent_type=agent_type
            )
            
            # Calculate reward
            reward = rl_agent.calculate_reward(metrics)
            
            # Create next state
            next_state = State(
                query=query,
                context=[msg['content'] for msg in history[:i+1]],
                previous_responses=[msg['content'] for msg in history[:i+1] if msg.get('role') == 'assistant'],
                agent_type=agent_type
            )
            
            # Train RL agent
            rl_agent.train(state, action, reward, next_state)
            
            # Add metrics to results
            evaluation_results['metrics'].append({
                'query': query,
                'response': response,
                'agent_type': agent_type,
                'metrics': metrics
            })
        
        # Calculate overall score
        if evaluation_results['metrics']:
            evaluation_results['overall_score'] = sum(
                sum(m['metrics'].values()) / len(m['metrics'])
                for m in evaluation_results['metrics']
            ) / len(evaluation_results['metrics'])
        
        # Get RL agent performance metrics
        evaluation_results['rl_metrics'] = rl_agent.get_performance_metrics()
        
        return evaluation_results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the API server
if __name__ == "__main__":
    uvicorn.run(
        "api.main:app", 
        host=API_HOST, 
        port=API_PORT,
        reload=True
    )