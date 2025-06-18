# Abstract classes

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from gaussian_payoff_graph import calculate_warmth_payoff


class PayoffCalculator:
    def gauss(
        self, my_warmth: float, other_warmth: float, alpha: float, beta: float
    ) -> float:
        """Calculate Gaussian payoff based on warmth values"""
        return calculate_warmth_payoff(
            w1=my_warmth, w2=other_warmth, alpha=alpha, beta=beta
        )


class AgentState(ABC):
    @abstractmethod
    def get_state_vector(self) -> np.ndarray:
        """Return the state vector representation of the agent"""
        pass

    @abstractmethod
    def update_state(self, my_action: float, other_action: float, payoff: float):
        """Update the agent's state based on the action taken and the received payoff"""
        pass

    @abstractmethod
    def reset_state(self):
        """Reset the agent's state to initial values"""
        pass

    @abstractmethod
    def get_state_dimension(self) -> int:
        """Return the dimension of the state vector"""
        pass


class AgentConfig(ABC):
    @abstractmethod
    def get_sac_params(self) -> Dict[str, Any]:
        """Return the parameters for the Soft Actor-Critic algorithm"""
        pass

    @abstractmethod
    def get_memory_length(self) -> int:
        """Return the length of the memory buffer for experience replay"""
        pass

    @abstractmethod
    def create_initial_state(self) -> AgentState:
        """Create and return an initial state for the agent"""
        pass


class ReinforcementLearner(ABC):
    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = True) -> float:
        """Select an action based on the current state"""
        pass

    @abstractmethod
    def store_transition(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store a transition in the agent's memory"""
        pass

    @abstractmethod
    def train_step(self) -> Optional[Dict[str, float]]:
        """Perform a training step and return training metrics if available"""
        pass

    @abstractmethod
    def save_model(self, filepath: str):
        """Save the agent's model to a file"""
        pass

    @abstractmethod
    def load_model(self, filepath: str):
        """Load the agent's model from a file"""
        pass


class InteractionEnvironment(ABC):
    @abstractmethod
    def step(
        self, my_action: float, other_action: float
    ) -> Tuple[np.ndarray, np.ndarray, float, float, bool]:
        """does one interaction step and returns the next state, reward, and done flag"""
        pass

    @abstractmethod
    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Reset the environment to its initial state and return the initial states of both agents"""
        pass

    @abstractmethod
    def get_state_dimension(self) -> int:
        """Return the dimension of the state space"""
        pass
