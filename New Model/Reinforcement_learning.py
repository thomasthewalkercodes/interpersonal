from dataclasses import dataclass
import numpy as np
from typing import Tuple, List
from abc import ABC, abstractmethod
from Payoff_matrix.gaussian_payoff import calculate_warmth_payoff


@dataclass
class LearningConfig:
    """Configuration for learning parameters"""

    alpha: float = 0.1  # Learning rate
    gamma: float = 0.95  # Discount factor
    epsilon: float = 0.1  # Exploration rate
    min_epsilon: float = 0.01  # Minimum exploration
    decay: float = 0.995  # Epsilon decay rate


class ActionSpace(ABC):
    """Abstract base class for different action spaces"""

    @abstractmethod
    def sample(self) -> float:
        """Return a random action"""
        pass

    @abstractmethod
    def discretize(self, action: float) -> int:
        """Convert continuous action to discrete state"""
        pass


class WarmthActionSpace(ActionSpace):
    """Handles warmth actions in [0,1] range"""

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.bins = np.linspace(0, 1, n_bins)

    def sample(self) -> float:
        return np.random.random()

    def discretize(self, action: float) -> int:
        return np.digitize(action, self.bins) - 1


class WarmthLearningAgent:
    """Q-learning agent for warmth-based interactions"""

    def __init__(
        self, config: LearningConfig, action_space: ActionSpace, name: str = "Agent"
    ):
        self.config = config
        self.action_space = action_space
        self.name = name
        self.epsilon = config.epsilon

        # Initialize Q-table
        self.q_table = np.zeros((action_space.n_bins, action_space.n_bins))

        # Learning history
        self.action_history: List[float] = []
        self.reward_history: List[float] = []

    def choose_action(self, state: int = None) -> float:
        """Choose action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            # Explore
            action = self.action_space.sample()
        else:
            # Exploit
            if state is None:
                state = 0
            action_idx = np.argmax(self.q_table[state])
            action = self.action_space.bins[action_idx]

        self.action_history.append(action)
        return action

    def update(self, state: int, action: float, reward: float, next_state: int):
        """Update Q-values using Q-learning"""
        action_idx = self.action_space.discretize(action)

        # Q-learning update
        old_value = self.q_table[state, action_idx]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.config.alpha) * old_value + self.config.alpha * (
            reward + self.config.gamma * next_max
        )

        self.q_table[state, action_idx] = new_value
        self.reward_history.append(reward)

        # Decay exploration rate
        self.epsilon = max(self.config.min_epsilon, self.epsilon * self.config.decay)


def interact(
    agent1: WarmthLearningAgent, agent2: WarmthLearningAgent
) -> Tuple[float, float]:
    """Simulate one interaction between agents"""
    # Get actions
    w1 = agent1.choose_action()
    w2 = agent2.choose_action()

    # Calculate payoffs
    payoff1 = calculate_warmth_payoff(w1, w2)
    payoff2 = calculate_warmth_payoff(w2, w1)  # Note the order swap

    # Update agents
    state1 = agent1.action_space.discretize(w1)
    state2 = agent2.action_space.discretize(w2)

    next_state1 = agent1.action_space.discretize(w2)  # Use other's action as next state
    next_state2 = agent2.action_space.discretize(w1)

    agent1.update(state1, w1, payoff1, next_state1)
    agent2.update(state2, w2, payoff2, next_state2)

    return payoff1, payoff2
