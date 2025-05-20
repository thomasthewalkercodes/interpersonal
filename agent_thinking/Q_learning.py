import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass
from Configurations import QLearningConfig
from .prior_handling import NashEquilibriumPrior


class QLearningAgent:
    """Q-learning agent with prospect theory"""

    def __init__(
        self,
        config: QLearningConfig,
        is_player1: bool,
        initial_strategy: float,
    ):
        """
        Initialize Q-learning agent
        Args:
            config: Learning parameters
            is_player1: True if this is Player 1 (using A1 matrix), False for Player 2 (using A2 matrix)
            initial_strategy: Nash equilibrium probability (p_star for Player 1, q_star for Player 2)
        """
        self.config = config
        self.is_player1 = is_player1
        self.prior_handler = NashEquilibriumPrior(initial_strategy)

        # Initialize Q-values based on player type
        if is_player1:
            self.Q_values = {"Up": 0.0, "Down": 0.0}
        else:
            self.Q_values = {"Left": 0.0, "Right": 0.0}

        self.reference_point = 0.0
        # Keep tracking for visualization
        self.current_round = 1
        self.action_counts = {k: 0 for k in self.Q_values.keys()}
        self.last_seen = {k: 0 for k in self.Q_values.keys()}

    def choose_action(self) -> str:
        """Select action using softmax policy with prior influence"""
        # Calculate basic softmax probabilities
        exp_q = {
            action: np.exp(self.config.beta * self.Q_values[action])
            for action in self.Q_values
        }
        total_exp_q = sum(exp_q.values())
        probs = {action: eq / total_exp_q for action, eq in exp_q.items()}

        # Use prior handler to blend probabilities
        if self.is_player1:
            prob_up = self.prior_handler.blend_probabilities(probs["Up"], self.config)
            return "Up" if np.random.random() < prob_up else "Down"
        else:
            prob_left = self.prior_handler.blend_probabilities(
                probs["Left"], self.config
            )
            return "Left" if np.random.random() < prob_left else "Right"

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

        # Keep action tracking for visualization
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
