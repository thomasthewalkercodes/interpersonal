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
    risk_sensitivity: float = 0.5  # How much to consider potential losses (0-1)
    prior_expectation: float = 0.5  # Expected warmth of other agent (0-1)
    prior_strength: float = 10.0  # Weight of prior beliefs (higher = stronger prior)


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

        # Initialize Q-table with prior expectations
        self.q_table = np.zeros((action_space.n_bins, action_space.n_bins))
        self._initialize_qtable_with_prior()

        # Track belief updates
        self.expected_warmth = config.prior_expectation
        self.belief_strength = config.prior_strength

        # Initialize history trackers
        self.action_history = []
        self.reward_history = []  # Add this line

    def _initialize_qtable_with_prior(self):
        """Initialize Q-values based on prior expectations"""
        for state in range(self.action_space.n_bins):
            for action_idx in range(self.action_space.n_bins):
                action = self.action_space.bins[action_idx]
                # Calculate expected value based on prior
                expected_other_action = self.config.prior_expectation
                expected_payoff = calculate_warmth_payoff(action, expected_other_action)
                self.q_table[state, action_idx] = (
                    expected_payoff * self.config.prior_strength
                )

    def choose_action(self, state: int = None) -> float:
        """Choose action using risk-aware epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            # Explore with bias towards medium warmth
            return np.random.beta(2, 2)  # Beta distribution centered around 0.5
        else:
            if state is None:
                state = 0

            # Calculate risk-adjusted values for each action
            action_values = []
            for i in range(self.action_space.n_bins):
                action = self.action_space.bins[i]
                q_value = self.q_table[state, i]

                # Penalize extreme warmth based on risk sensitivity
                risk_penalty = self.config.risk_sensitivity * abs(action - 0.5)
                adjusted_value = q_value - risk_penalty

                action_values.append(adjusted_value)

            # Choose action with highest risk-adjusted value
            best_action_idx = np.argmax(action_values)
            return self.action_space.bins[best_action_idx]

    def update(self, state: int, action: float, reward: float, next_state: int):
        """Update Q-values using Q-learning with prior influence"""
        action_idx = self.action_space.discretize(action)

        # Calculate learning rate based on prior strength
        effective_alpha = self.config.alpha / (1 + self.belief_strength / 100)

        # Update expected warmth using exponential moving average
        other_action = self.action_space.bins[next_state]
        self.expected_warmth = (
            1 - effective_alpha
        ) * self.expected_warmth + effective_alpha * other_action

        # Gradually reduce prior strength as we get more experience
        self.belief_strength = max(1.0, self.belief_strength * 0.995)

        # Q-learning update with prior influence
        old_value = self.q_table[state, action_idx]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - effective_alpha) * old_value + effective_alpha * (
            reward + self.config.gamma * next_max
        )

        self.q_table[state, action_idx] = new_value
        self.reward_history.append(reward)

        # Decay exploration rate
        self.epsilon = max(self.config.min_epsilon, self.epsilon * self.config.decay)
