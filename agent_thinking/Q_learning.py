import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass
from Configurations import QLearningConfig, A1, A2  # Add matrices to import
from .prior_handling import NashEquilibriumPrior


class QLearningAgent:
    """Q-learning agent with prospect theory"""

    def __init__(
        self,
        config: QLearningConfig,
        is_player1: bool,
        initial_strategy: float,
    ):
        """Initialize Q-learning agent with payoff-based Q-values"""
        self.config = config
        self.is_player1 = is_player1
        self.prior_handler = NashEquilibriumPrior(initial_strategy)

        # Calculate initial Q-values based on expected payoffs under Nash equilibrium
        if is_player1:
            payoff_matrix = A1
            expected_up = payoff_matrix[0, 0] * initial_strategy + payoff_matrix[
                0, 1
            ] * (1 - initial_strategy)
            expected_down = payoff_matrix[1, 0] * initial_strategy + payoff_matrix[
                1, 1
            ] * (1 - initial_strategy)
            self.Q_values = {"Up": expected_up, "Down": expected_down}
        else:
            payoff_matrix = A2
            expected_left = payoff_matrix[0, 0] * initial_strategy + payoff_matrix[
                1, 0
            ] * (1 - initial_strategy)
            expected_right = payoff_matrix[0, 1] * initial_strategy + payoff_matrix[
                1, 1
            ] * (1 - initial_strategy)
            self.Q_values = {"Left": expected_left, "Right": expected_right}

        self.reference_point = 0.0
        self.current_round = 1
        self.action_counts = {k: 0 for k in self.Q_values.keys()}
        self.last_seen = {k: 0 for k in self.Q_values.keys()}

    def choose_action(self) -> str:
        """Select action using softmax policy with prior influence"""
        # Calculate probabilities with higher beta for more exploitation
        exp_q = {
            action: np.exp(self.config.beta * self.Q_values[action])
            for action in self.Q_values
        }
        total_exp_q = sum(exp_q.values())
        probs = {action: eq / total_exp_q for action, eq in exp_q.items()}

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
        # Direct payoff update without risk aversion for clearer learning
        value = payoff  # Remove complexity of prospect theory initially

        # Standard Q-learning update
        old_value = self.Q_values[action]
        self.Q_values[action] = old_value + self.config.alpha * (value - old_value)


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
