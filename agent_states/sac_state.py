"""
Agent state implementation for interpersonal simulations.

This module implements the agent state management system that tracks
psychological variables like trust, satisfaction, and interaction history.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
from interfaces import AgentState


class InterpersonalAgentState(AgentState):
    """
    Implementation of agent state for interpersonal interactions.
    
    This class manages the psychological state of an agent including:
    - Trust levels towards other agents
    - Satisfaction with recent interactions
    - Memory of past interactions
    - Emotional state and arousal
    """
    
    def __init__(self, 
                 memory_length: int = 50,
                 initial_trust: float = 0.0,
                 initial_satisfaction: float = 0.0,
                 trust_learning_rate: float = 0.1,
                 satisfaction_decay: float = 0.95):
        """
        Initialize the interpersonal agent state.
        
        Args:
            memory_length: Number of past interactions to remember
            initial_trust: Starting trust level [-1, 1]
            initial_satisfaction: Starting satisfaction level [-1, 1]
            trust_learning_rate: How quickly trust updates based on outcomes
            satisfaction_decay: How quickly satisfaction decays over time
        """
        self.memory_length = memory_length
        self.trust_learning_rate = trust_learning_rate
        self.satisfaction_decay = satisfaction_decay
        
        # Core psychological variables
        self.trust = initial_trust
        self.satisfaction = initial_satisfaction
        self.arousal = 0.0  # Emotional arousal level
        
        # Interaction memory
        self.action_memory = deque(maxlen=memory_length)
        self.other_action_memory = deque(maxlen=memory_length)
        self.reward_memory = deque(maxlen=memory_length)
        self.outcome_memory = deque(maxlen=memory_length)  # Success/failure of interactions
        
        # Statistics
        self.interaction_count = 0
        self.successful_interactions = 0
        self.average_reward = 0.0
        
        # Initialize memory with neutral values
        self.reset()
    
    def update(self, action: float, other_action: float, reward: float) -> None:
        """
        Update the agent's psychological state based on interaction outcome.
        
        Args:
            action: The action this agent took [-1, 1]
            other_action: The action the other agent took [-1, 1]
            reward: The reward received for this interaction
        """
        # Store in memory
        self.action_memory.append(action)
        self.other_action_memory.append(other_action)
        self.reward_memory.append(reward)
        
        # Determine if this was a "successful" interaction (above average)
        success = reward > self.average_reward if self.interaction_count > 0 else reward > 0
        self.outcome_memory.append(1.0 if success else -1.0)
        
        # Update trust based on interaction outcome and other's behavior
        self._update_trust(other_action, reward, success)
        
        # Update satisfaction based on reward
        self._update_satisfaction(reward)
        
        # Update arousal based on surprise (deviation from expectation)
        self._update_arousal(reward)
        
        # Update statistics
        self.interaction_count += 1
        if success:
            self.successful_interactions += 1
        
        # Update running average reward
        alpha = min(0.1, 1.0 / self.interaction_count)  # Adaptive learning rate
        self.average_reward = (1 - alpha) * self.average_reward + alpha * reward
    
    def _update_trust(self, other_action: float, reward: float, success: bool) -> None:
        """Update trust level based on other agent's behavior and outcomes."""
        # Convert other_action from [-1,1] to [0,1] for warmth interpretation
        other_warmth = (other_action + 1) / 2
        
        # Trust increases when:
        # 1. Other agent shows warmth (high other_warmth)
        # 2. Interaction was successful
        # 3. Other agent's behavior was predictable (consistent with history)
        
        warmth_factor = other_warmth  # Direct warmth contribution
        success_factor = 0.5 if success else -0.3  # Success/failure impact
        
        # Predictability factor - how consistent is other agent?
        predictability_factor = 0.0
        if len(self.other_action_memory) > 1:
            recent_actions = list(self.other_action_memory)[-5:]  # Last 5 actions
            if len(recent_actions) > 1:
                consistency = 1.0 - np.std(recent_actions)  # Lower std = more consistent
                predictability_factor = 0.2 * consistency
        
        # Combine factors
        trust_delta = self.trust_learning_rate * (
            0.4 * warmth_factor + 
            0.4 * success_factor + 
            0.2 * predictability_factor
        )
        
        # Update trust with bounds
        self.trust = np.clip(self.trust + trust_delta, -1.0, 1.0)
    
    def _update_satisfaction(self, reward: float) -> None:
        """Update satisfaction level based on reward received."""
        # Satisfaction decays over time
        self.satisfaction *= self.satisfaction_decay
        
        # Add current reward impact
        reward_impact = 0.3 * (reward - 0.5)  # Normalize around neutral point
        self.satisfaction = np.clip(self.satisfaction + reward_impact, -1.0, 1.0)
    
    def _update_arousal(self, reward: float) -> None:
        """Update arousal based on surprise/prediction error."""
        if self.interaction_count > 0:
            prediction_error = abs(reward - self.average_reward)
            max_possible_error = 2.0  # Assuming rewards roughly in [-1, 1] range
            
            # Higher prediction error = higher arousal
            target_arousal = min(1.0, prediction_error / max_possible_error)
            
            # Smooth arousal changes
            arousal_lr = 0.3
            self.arousal = (1 - arousal_lr) * self.arousal + arousal_lr * target_arousal
        else:
            self.arousal = 0.5  # Moderate arousal for first interaction
    
    def get_state_vector(self) -> np.ndarray:
        """
        Get the current state as a vector for neural network input.
        
        Returns:
            State vector: [own_last_action, other_last_action, trust, satisfaction, 
                          arousal, avg_other_warmth, interaction_success_rate]
        """
        # Get last actions (or 0 if no history)
        own_last_action = self.action_memory[-1] if self.action_memory else 0.0
        other_last_action = self.other_action_memory[-1] if self.other_action_memory else 0.0
        
        # Calculate average other warmth
        if self.other_action_memory:
            # Convert from [-1,1] to [0,1] for warmth interpretation
            other_warmths = [(a + 1) / 2 for a in self.other_action_memory]
            avg_other_warmth = np.mean(other_warmths)
        else:
            avg_other_warmth = 0.5  # Neutral expectation
        
        # Calculate success rate
        success_rate = (self.successful_interactions / max(1, self.interaction_count))
        
        # Compile state vector
        state = np.array([
            own_last_action,      # 0: Own last action
            other_last_action,    # 1: Other's last action  
            self.trust,           # 2: Trust level
            self.satisfaction,    # 3: Satisfaction level
            self.arousal,         # 4: Arousal level
            avg_other_warmth,     # 5: Average perceived warmth of other
            success_rate          # 6: Success rate in interactions
        ], dtype=np.float32)
        
        return state
    
    def get_trust_level(self) -> float:
        """Get the current trust level towards the other agent."""
        return self.trust
    
    def get_satisfaction_level(self) -> float:
        """Get the current satisfaction level."""
        return self.satisfaction
    
    def get_arousal_level(self) -> float:
        """Get the current arousal level."""
        return self.arousal
    
    def get_interaction_history(self) -> Dict[str, List[float]]:
        """
        Get the full interaction history.
        
        Returns:
            Dictionary with action and reward histories
        """
        return {
            'own_actions': list(self.action_memory),
            'other_actions': list(self.other_action_memory),
            'rewards': list(self.reward_memory),
            'outcomes': list(self.outcome_memory)
        }
    
    def get_memory_summary(self) -> Dict[str, float]:
        """
        Get summary statistics of the agent's memory.
        
        Returns:
            Dictionary with memory statistics
        """
        if not self.reward_memory:
            return {
                'avg_reward': 0.0,
                'reward_std': 0.0,
                'avg_own_warmth': 0.5,
                'avg_other_warmth': 0.5,
                'success_rate': 0.0,
                'memory_length': 0
            }
        
        # Convert actions to warmth levels for interpretation
        own_warmths = [(a + 1) / 2 for a in self.action_memory]
        other_warmths = [(a + 1) / 2 for a in self.other_action_memory]
        
        return {
            'avg_reward': np.mean(self.reward_memory),
            'reward_std': np.std(self.reward_memory),
            'avg_own_warmth': np.mean(own_warmths),
            'avg_other_warmth': np.mean(other_warmths),
            'success_rate': self.successful_interactions / max(1, self.interaction_count),
            'memory_length': len(self.reward_memory)
        }
    
    def predict_other_action(self) -> float:
        """
        Predict the other agent's next action based on history.
        
        Returns:
            Predicted action (simple moving average of recent actions)
        """
        if len(self.other_action_memory) == 0:
            return 0.0  # Neutral prediction
        
        # Use recent history for prediction (last 5 actions)
        recent_actions = list(self.other_action_memory)[-5:]
        return np.mean(recent_actions)
    
    def reset(self) -> None:
        """Reset the agent state to initial conditions."""
        # Clear memory
        self.action_memory.clear()
        self.other_action_memory.clear()
        self.reward_memory.clear()
        self.outcome_memory.clear()
        
        # Reset psychological variables to initial values
        self.trust = 0.0  # Will be set by config
        self.satisfaction = 0.0  # Will be set by config
        self.arousal = 0.0
        
        # Reset statistics
        self.interaction_count = 0
        self.successful_interactions = 0
        self.average_reward = 0.0
        
        # Fill memory with neutral values to maintain consistent state size
        neutral_action = 0.0
        for _ in range(min(3, self.memory_length)):  # Start with small memory
            self.action_memory.append(neutral_action)
            self.other_action_memory.append(neutral_action)
            self.reward_memory.append(0.0)
            self.outcome_memory.append(0.0)
    
    def get_state_dim(self) -> int:
        """Get the dimensionality of the state vector."""
        return 7  # Length of state vector from get_state_vector()
    
    def is_trusting(self) -> bool:
        """Check if agent currently trusts the other agent."""
        return self.trust > 0.1
    
    def is_satisfied(self) -> bool:
        """Check if agent is currently satisfied with interactions."""
        return self.satisfaction > 0.1
    
    def is_aroused(self) -> bool:
        """Check if agent is currently in high arousal state."""
        return self.arousal > 0.7
    
    def get_emotional_state(self) -> str:
        """
        Get a descriptive label for the current emotional state.
        
        Returns:
            String describing the emotional state
        """
        if self.satisfaction > 0.3 and self.arousal < 0.4:
            return "content"
        elif self.satisfaction > 0.3 and self.arousal > 0.6:
            return "excited"
        elif self.satisfaction < -0.3 and self.arousal > 0.6:
            return "frustrated"
        elif self.satisfaction < -0.3 and self.arousal < 0.4:
            return "disappointed"
        elif self.trust > 0.5:
            return "trusting"
        elif self.trust < -0.5:
            return "suspicious"
        else:
            return "neutral"
    
    def __str__(self) -> str:
        """String representation of the agent state."""
        return (f"InterpersonalAgentState("
                f"trust={self.trust:.3f}, "
                f"satisfaction={self.satisfaction:.3f}, "
                f"arousal={self.arousal:.3f}, "
                f"interactions={self.interaction_count}, "
                f"emotion='{self.get_emotional_state()}')")
    
    def __repr__(self) -> str:
        """Detailed representation of the agent state."""
        return self.__str__()


class GroupAgentState(AgentState):
    """
    Extended agent state for group interactions (3+ agents).
    
    This class manages state for agents that interact with multiple others
    simultaneously, tracking relationships with each individual agent.
    """
    
    def __init__(self, 
                 num_other_agents: int,
                 memory_length: int = 50,
                 initial_trust: float = 0.0,
                 initial_satisfaction: float = 0.0):
        """
        Initialize group agent state.
        
        Args:
            num_other_agents: Number of other agents in the group
            memory_length: Memory buffer size
            initial_trust: Starting trust level
            initial_satisfaction: Starting satisfaction level
        """
        self.num_other_agents = num_other_agents
        
        # Create separate state tracking for each other agent
        self.individual_states = {}
        for i in range(num_other_agents):
            self.individual_states[f"agent_{i}"] = InterpersonalAgentState(
                memory_length=memory_length,
                initial_trust=initial_trust,
                initial_satisfaction=initial_satisfaction
            )
        
        # Global group-level variables
        self.group_cohesion = 0.0
        self.social_status = 0.0  # Perceived status within group
        self.group_satisfaction = 0.0
    
    def get_state_dim(self) -> int:
        """Get dimensionality of group state vector."""
        return 10  # 7 from individual + 3 group features


class AdaptiveAgentState(InterpersonalAgentState):
    """
    Enhanced agent state with adaptive learning mechanisms.
    
    This class extends the basic interpersonal state with more sophisticated
    learning and adaptation capabilities based on interaction patterns.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Adaptive learning parameters
        self.learning_rate_adaptation = True
        self.base_trust_lr = self.trust_learning_rate
        self.trust_lr_range = (0.01, 0.3)
        
        # Pattern recognition
        self.cooperation_streak = 0
        self.defection_streak = 0
        self.pattern_memory = deque(maxlen=10)  # Remember behavioral patterns
        
        # Meta-learning variables
        self.exploration_tendency = 0.5  # How much to explore vs exploit
        self.risk_tolerance = 0.5  # Willingness to take risks
    
    def update(self, action: float, other_action: float, reward: float) -> None:
        """Enhanced update with adaptive learning."""
        # Call parent update
        super().update(action, other_action, reward)
        
        # Update adaptive mechanisms
        self._update_learning_rates(reward)
        self._update_behavioral_patterns(other_action)
        self._update_meta_learning(action, other_action, reward)
    
    def _update_learning_rates(self, reward: float) -> None:
        """Adapt learning rates based on performance."""
        if not self.learning_rate_adaptation:
            return
        
        # Increase learning rate if performance is poor, decrease if good
        performance_signal = reward - self.average_reward if self.interaction_count > 5 else 0
        
        if performance_signal < -0.1:  # Poor performance
            self.trust_learning_rate = min(self.trust_lr_range[1], 
                                         self.trust_learning_rate * 1.1)
        elif performance_signal > 0.1:  # Good performance
            self.trust_learning_rate = max(self.trust_lr_range[0],
                                         self.trust_learning_rate * 0.95)
    
    def _update_behavioral_patterns(self, other_action: float) -> None:
        """Track and recognize patterns in other agent's behavior."""
        other_warmth = (other_action + 1) / 2
        
        # Track cooperation/defection streaks
        if other_warmth > 0.6:  # Cooperative behavior
            self.cooperation_streak += 1
            self.defection_streak = 0
        elif other_warmth < 0.4:  # Uncooperative behavior
            self.defection_streak += 1
            self.cooperation_streak = 0
        else:  # Neutral behavior
            self.cooperation_streak = 0
            self.defection_streak = 0
        
        # Store pattern information
        pattern_info = {
            'action': other_action,
            'warmth': other_warmth,
            'cooperation_streak': self.cooperation_streak,
            'defection_streak': self.defection_streak
        }
        self.pattern_memory.append(pattern_info)
    
    def _update_meta_learning(self, action: float, other_action: float, reward: float) -> None:
        """Update meta-learning parameters."""
        own_warmth = (action + 1) / 2
        other_warmth = (other_action + 1) / 2
        
        # Update exploration tendency based on reward relative to risk taken
        risk_taken = own_warmth  # Higher warmth = higher risk
        if risk_taken > 0.1:  # Only update if some risk was taken
            risk_reward_ratio = reward / risk_taken
            if risk_reward_ratio > 1.0:  # Risk paid off
                self.exploration_tendency = min(1.0, self.exploration_tendency + 0.05)
            else:  # Risk didn't pay off
                self.exploration_tendency = max(0.0, self.exploration_tendency - 0.03)
        
        # Update risk tolerance based on outcomes
        if reward > self.average_reward:
            self.risk_tolerance = min(1.0, self.risk_tolerance + 0.02)
        else:
            self.risk_tolerance = max(0.0, self.risk_tolerance - 0.01)
    
    def get_state_vector(self) -> np.ndarray:
        """Enhanced state vector with adaptive features."""
        base_state = super().get_state_vector()
        
        # Add adaptive features
        adaptive_features = np.array([
            self.cooperation_streak / 10.0,  # Normalized cooperation streak
            self.defection_streak / 10.0,    # Normalized defection streak
            self.exploration_tendency,       # Current exploration tendency
            self.risk_tolerance,             # Current risk tolerance
            self.trust_learning_rate / 0.3  # Normalized learning rate
        ])
        
        return np.concatenate([base_state, adaptive_features])
    
    def get_state_dim(self) -> int:
        """Get dimensionality of adaptive state vector."""
        return 12  # 7 base + 5 adaptive features
    
    def should_explore(self) -> bool:
        """Decide whether to explore or exploit based on current state."""
        # Explore more when:
        # 1. High exploration tendency
        # 2. Low satisfaction (need to try something different)
        # 3. High arousal (uncertainty suggests exploration)
        
        explore_score = (
            0.4 * self.exploration_tendency +
            0.3 * max(0, -self.satisfaction) +  # Dissatisfaction encourages exploration
            0.3 * self.arousal
        )
        
        return explore_score > 0.5
    
    def get_pattern_prediction(self) -> Optional[str]:
        """
        Predict the other agent's behavioral pattern.
        
        Returns:
            String describing predicted pattern or None if no clear pattern
        """
        if len(self.pattern_memory) < 5:
            return None
        
        recent_patterns = list(self.pattern_memory)[-5:]
        
        # Check for consistent cooperation
        if all(p['cooperation_streak'] > 0 for p in recent_patterns):
            return "cooperative"
        
        # Check for consistent defection
        if all(p['defection_streak'] > 0 for p in recent_patterns):
            return "uncooperative"
        
        # Check for alternating pattern
        warmths = [p['warmth'] for p in recent_patterns]
        if len(set(np.round(warmths, 1))) > 3:  # High variance
            return "unpredictable"
        
        return "neutral"


def create_agent_state(state_type: str = "basic", **kwargs) -> AgentState:
    """
    Factory function for creating different types of agent states.
    
    Args:
        state_type: Type of state to create ('basic', 'adaptive', 'group')
        **kwargs: Additional parameters for state initialization
        
    Returns:
        AgentState instance of the specified type
    """
    if state_type == "basic":
        return InterpersonalAgentState(**kwargs)
    elif state_type == "adaptive":
        return AdaptiveAgentState(**kwargs)
    elif state_type == "group":
        return GroupAgentState(**kwargs)
    else:
        raise ValueError(f"Unknown state type: {state_type}")


def analyze_state_evolution(state_history: List[AgentState]) -> Dict[str, Any]:
    """
    Analyze how an agent's state evolved over time.
    
    Args:
        state_history: List of agent states from different time points
        
    Returns:
        Dictionary containing evolution analysis
    """
    if not state_history:
        return {}
    
    # Extract time series of key variables
    trust_evolution = [state.get_trust_level() for state in state_history]
    satisfaction_evolution = [state.satisfaction for state in state_history]
    arousal_evolution = [state.arousal for state in state_history]
    
    # Calculate trends
    trust_trend = np.polyfit(range(len(trust_evolution)), trust_evolution, 1)[0]
    satisfaction_trend = np.polyfit(range(len(satisfaction_evolution)), satisfaction_evolution, 1)[0]
    
    # Calculate volatility
    trust_volatility = np.std(trust_evolution)
    satisfaction_volatility = np.std(satisfaction_evolution)
    
    # Identify key turning points
    trust_changes = np.diff(trust_evolution)
    major_trust_changes = np.where(np.abs(trust_changes) > 0.2)[0]
    
    return {
        'trust_trend': trust_trend,
        'satisfaction_trend': satisfaction_trend,
        'trust_volatility': trust_volatility,
        'satisfaction_volatility': satisfaction_volatility,
        'major_trust_changes': major_trust_changes.tolist(),
        'final_trust': trust_evolution[-1],
        'final_satisfaction': satisfaction_evolution[-1],
        'avg_arousal': np.mean(arousal_evolution),
        'state_evolution_summary': {
            'trust': {'start': trust_evolution[0], 'end': trust_evolution[-1], 'trend': trust_trend},
            'satisfaction': {'start': satisfaction_evolution[0], 'end': satisfaction_evolution[-1], 'trend': satisfaction_trend}
        }
    }initial_satisfaction
    
    def update(self, action: float, other_actions: Dict[str, float], rewards: Dict[str, float]) -> None:
        """
        Update state based on group interaction.
        
        Args:
            action: This agent's action
            other_actions: Dictionary mapping agent_id -> action
            rewards: Dictionary mapping agent_id -> reward received from that agent
        """
        # Update individual relationships
        for agent_id, other_action in other_actions.items():
            if agent_id in self.individual_states:
                reward = rewards.get(agent_id, 0.0)
                self.individual_states[agent_id].update(action, other_action, reward)
        
        # Update group-level metrics
        self._update_group_metrics(other_actions, rewards)
    
    def _update_group_metrics(self, other_actions: Dict[str, float], rewards: Dict[str, float]) -> None:
        """Update group-level psychological variables."""
        # Group cohesion based on similarity of actions
        if len(other_actions) > 0:
            action_variance = np.var(list(other_actions.values()))
            self.group_cohesion = max(0.0, 1.0 - action_variance)  # High cohesion = low variance
        
        # Social status based on relative performance
        avg_reward = np.mean(list(rewards.values())) if rewards else 0.0
        self.social_status = np.tanh(avg_reward)  # Normalize to [-1, 1]
        
        # Group satisfaction
        self.group_satisfaction = 0.9 * self.group_satisfaction + 0.1 * avg_reward
    
    def get_state_vector(self) -> np.ndarray:
        """Get state vector incorporating group dynamics."""
        # Get individual state from primary relationship (agent_0)
        if "agent_0" in self.individual_states:
            individual_state = self.individual_states["agent_0"].get_state_vector()
        else:
            individual_state = np.zeros(7)
        
        # Add group-specific features
        group_features = np.array([
            self.group_cohesion,
            self.social_status,
            self.group_satisfaction
        ])
        
        return np.concatenate([individual_state, group_features])
    
    def get_trust_level(self, agent_id: Optional[str] = None) -> float:
        """Get trust level (group average or specific agent)."""
        if agent_id and agent_id in self.individual_states:
            return self.individual_states[agent_id].get_trust_level()
        
        # Return average trust across all relationships
        if self.individual_states:
            trust_levels = [state.get_trust_level() for state in self.individual_states.values()]
            return np.mean(trust_levels)
        return 0.0
    
    def reset(self) -> None:
        """Reset all individual and group states."""
        for state in self.individual_states.values():
            state.reset()
        
        self.group_cohesion = 0.0
        self.social_status = 0.0
        self.group_satisfaction =