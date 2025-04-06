import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import json
import os
from datetime import datetime

@dataclass
class State:
    """Represents the state of a conversation"""
    query: str
    context: List[str]
    previous_responses: List[str]
    agent_type: str

@dataclass
class Action:
    """Represents an action taken by the agent"""
    response: str
    confidence: float
    agent_type: str

@dataclass
class Reward:
    """Represents the reward for an action"""
    coherence: float
    relevance: float
    helpfulness: float
    accuracy:float
    user_satisfaction:float
    total: float
    bonus:float

class RLAgent:
    """Reinforcement Learning agent for improving chatbot responses"""
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = {}
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        
        # Load existing Q-table if available
        self.load_q_table()
    
    def get_state_key(self, state: State) -> str:
        """Convert state to a hashable key for Q-table"""
        return json.dumps({
            'query': state.query,
            'context': state.context,
            'agent_type': state.agent_type
        })
    
    def get_action_key(self, action: Action) -> str:
        """Convert action to a hashable key for Q-table"""
        return json.dumps({
            'response': action.response,
            'confidence': action.confidence,
            'agent_type': action.agent_type
        })
    
    def get_q_value(self, state: State, action: Action) -> float:
        """Get Q-value for a state-action pair"""
        state_key = self.get_state_key(state)
        action_key = self.get_action_key(action)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        if action_key not in self.q_table[state_key]:
            self.q_table[state_key][action_key] = 0.0
        
        return self.q_table[state_key][action_key]
    
    def update_q_value(self, state: State, action: Action, reward: float, next_state: State):
        """Update Q-value using Q-learning algorithm"""
        state_key = self.get_state_key(state)
        action_key = self.get_action_key(action)
        
        # Get current Q-value
        current_q = self.get_q_value(state, action)
        
        # Get max Q-value for next state
        next_state_key = self.get_state_key(next_state)
        if next_state_key in self.q_table:
            next_max_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0
        else:
            next_max_q = 0
        
        # Q-learning update formula
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )
        
        # Update Q-table
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        self.q_table[state_key][action_key] = new_q
    
    def choose_action(self, state: State, available_actions: List[Action]) -> Action:
        """Choose an action using epsilon-greedy strategy"""
        epsilon = 0.1  # Exploration rate
        
        if np.random.random() < epsilon:
            # Explore: choose random action
            return np.random.choice(available_actions)
        else:
            # Exploit: choose best action
            state_key = self.get_state_key(state)
            if state_key in self.q_table:
                action_values = {
                    self.get_action_key(action): self.get_q_value(state, action)
                    for action in available_actions
                }
                best_action_key = max(action_values.items(), key=lambda x: x[1])[0]
                return next(
                    action for action in available_actions
                    if self.get_action_key(action) == best_action_key
                )
            else:
                return np.random.choice(available_actions)
    
    def calculate_reward(self, metrics: Dict[str, float]) -> Reward:
        """Calculate reward based on evaluation metrics"""
        coherence = metrics.get('coherence', 0.0)
        relevance = metrics.get('relevance', 0.0)
        helpfulness = metrics.get('helpfulness', 0.0)
        accuracy = metrics.get('accuracy', 0.0)
        user_satisfaction = metrics.get('user_satisfactionc', 0.0)
        
        # Updated weights based on importance
        weights = {
            'accuracy': 0.25,       # Factual correctness is crucial
            'coherence': 0.20,      # Logical flow matters
            'relevance': 0.25,      # Must address the query
            'helpfulness': 0.15,    # Practical utility
            'user_satisfaction': 0.15  # User experience
        }
        
        # Calculate weighted total with non-linear bonuses
        base_reward = (
            weights['accuracy'] * accuracy +
            weights['coherence'] * coherence +
            weights['relevance'] * relevance +
            weights['helpfulness'] * helpfulness +
            weights['user_satisfaction'] * user_satisfaction
        )
        
        # Add bonuses for excellent performance in key areas
        bonus = 0
        if accuracy > 0.9:
            bonus += 0.05  # High accuracy bonus
        if user_satisfaction > 0.85:
            bonus += 0.03  # Delighted users bonus
        if all(v > 0.7 for v in [accuracy, relevance]):
            bonus += 0.02  # Core competency bonus
            
        total = min(base_reward + bonus, 1.0)  # Cap at 1.0
        
        return Reward(
            accuracy=accuracy,
            coherence=coherence,
            relevance=relevance,
            helpfulness=helpfulness,
            user_satisfaction=user_satisfaction,
            total=total,
            bonus=bonus
        )
        
    def save_q_table(self):
        """Save Q-table to file"""
        os.makedirs('evaluation/data', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'evaluation/data/q_table_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(self.q_table, f)
    
    def load_q_table(self):
        """Load Q-table from file"""
        try:
            # Get the most recent Q-table file
            q_table_files = [f for f in os.listdir('evaluation/data') if f.startswith('q_table_')]
            if q_table_files:
                latest_file = max(q_table_files)
                with open(f'evaluation/data/{latest_file}', 'r') as f:
                    self.q_table = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.q_table = {}
    
    def train(self, state: State, action: Action, reward: Reward, next_state: State):
        """Train the RL agent with a new experience"""
        # Store experience
        self.state_history.append(state)
        self.action_history.append(action)
        self.reward_history.append(reward)
        
        # Update Q-values
        self.update_q_value(state, action, reward.total, next_state)
        
        # Periodically save Q-table
        if len(self.state_history) % 100 == 0:
            self.save_q_table()
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for the RL agent"""
        if not self.reward_history:
            return {
                'average_reward': 0.0,
                'average_coherence': 0.0,
                'average_relevance': 0.0,
                'average_helpfulness': 0.0
            }
        
        return {
            'average_reward': np.mean([r.total for r in self.reward_history]),
            'average_coherence': np.mean([r.coherence for r in self.reward_history]),
            'average_relevance': np.mean([r.relevance for r in self.reward_history]),
            'average_helpfulness': np.mean([r.helpfulness for r in self.reward_history])
        } 