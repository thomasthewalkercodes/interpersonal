# Q-Learning (machine learning)
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class QLearningConfig:
    """Configuration parameters for Q-learning agent"""

    alpha: float  # Learning rate
    beta: float  # Temperature parameter for softmax
    gamma: float  # Discount factor
    rho: float  # Risk aversion parameter
    lambda_val: float  # Loss aversion parameter
    ema_weight: float  # Weight for exponential moving average
    novelty_weight: float  # Weight for novelty bonus
    novelty_decay: float  # Decay rate for novelty bonus
    prior_weight: float  # Weight for prior knowledge
    rand_explore: float  # Random exploration probability


class QLearningAgent:
    """Q-learning agent with prospect theory and novelty seeking"""

    def __init__(self, config: QLearningConfig, initial_strategy: float):
        self.config = config
        self.Q_values = {"Up": 0.0, "Down": 0.0}  # Q-values for actions
        self.reference_point = 0.0  # Reference point for prospect theory
        self.last_action = None
        self.action_counts = {"Up": 0, "Down": 0}
        self.last_seen = {"Up": 0, "Down": 0}
        self.current_round = 1
        self.prior_probability = initial_strategy

    def get_novelty_bonus(self, action: str) -> float:
        """Calculate novelty bonus for an action"""
        rounds_since_seen = self.current_round - self.last_seen[action]
        return rounds_since_seen**self.config.novelty_decay

    def choose_action(self, explore: bool = True) -> str:
        """Select action using softmax policy with exploration"""
        if explore and np.random.random() < self.config.rand_explore:
            return np.random.choice(["Up", "Down"])

        # Combine learned values with prior knowledge
        novelty_bonus = {
            action: self.get_novelty_bonus(action) for action in self.Q_values
        }

        novelty_mean = np.mean(list(novelty_bonus.values()))
        current_prior_weight = (
            self.config.prior_weight + self.config.novelty_weight * novelty_mean
        )

        # Softmax probability calculation
        learned_probs = self._calculate_softmax_probabilities()
        final_prob_up = (1 - current_prior_weight) * learned_probs[
            "Up"
        ] + current_prior_weight * self.prior_probability

        return "Up" if np.random.random() < final_prob_up else "Down"

    def _calculate_softmax_probabilities(self) -> Dict[str, float]:
        """Calculate action probabilities using softmax"""
        exp_q_up = np.exp(self.config.beta * self.Q_values["Up"])
        exp_q_down = np.exp(self.config.beta * self.Q_values["Down"])
        prob_up = exp_q_up / (exp_q_up + exp_q_down)
        return {"Up": prob_up, "Down": 1 - prob_up}

    def update(self, action: str, payoff: float) -> None:
        """Update Q-values and reference point based on received payoff"""
        # Risk aversion transformation
        utility = np.sign(payoff) * abs(payoff) ** self.config.rho

        # Prospect theory value calculation
        if utility >= self.reference_point:
            value = (utility - self.reference_point) ** self.config.gamma
        else:
            value = (
                -self.config.lambda_val
                * (self.reference_point - utility) ** self.config.gamma
            )

        # Q-value update
        self.Q_values[action] += self.config.alpha * (value - self.Q_values[action])

        # Update reference point using exponential moving average
        self.reference_point = (
            self.config.ema_weight * utility
            + (1 - self.config.ema_weight) * self.reference_point
        )

        # Update action tracking
        self.last_seen[action] = self.current_round
        self.action_counts[action] += 1
        self.current_round += 1


class GameEnvironment:
    """Environment for two-player game with Q-learning agents"""

    def __init__(
        self,
        payoff_matrix1: np.ndarray,
        payoff_matrix2: np.ndarray,
        agent1: QLearningAgent,
        agent2: QLearningAgent,
    ):
        self.payoff_matrix1 = payoff_matrix1
        self.payoff_matrix2 = payoff_matrix2
        self.agent1 = agent1
        self.agent2 = agent2

    def step(self) -> Tuple[str, str, float, float]:
        """Execute one step of the game"""
        action1 = self.agent1.choose_action()
        action2 = self.agent2.choose_action()

        idx1 = 0 if action1 == "Up" else 1
        idx2 = 0 if action2 == "Left" else 1

        payoff1 = self.payoff_matrix1[idx1, idx2]
        payoff2 = self.payoff_matrix2[idx1, idx2]

        self.agent1.update(action1, payoff1)
        self.agent2.update(action2, payoff2)

        return action1, action2, payoff1, payoff2
