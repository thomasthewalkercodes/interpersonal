import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass
from Configurations import QLearningConfig, A1, A2
from .prior_handling import NashEquilibriumPrior
import nashpy as nash


def get_best_equilibrium(equilibria: List[Tuple]) -> Tuple:
    """Select mixed equilibrium if it exists, otherwise take first pure equilibrium"""
    for p, q in equilibria:
        # Check if this is a mixed strategy (not all probabilities are 0 or 1)
        if not all(x in [0.0, 1.0] for x in np.concatenate([p, q])):
            return p, q
    # If no mixed strategy found, return first equilibrium
    return equilibria[0]


class QLearningAgent:
    """Q-learning agent with prospect theory"""

    def __init__(
        self,
        config: QLearningConfig,
        is_player1: bool,
        payoff_matrices: Tuple[np.ndarray, np.ndarray],
    ):
        """Initialize Q-learning agent with Nash equilibrium strategies"""
        self.config = config
        self.is_player1 = is_player1

        # Calculate Nash equilibrium
        game = nash.Game(payoff_matrices[0], payoff_matrices[1])
        equilibria = list(game.support_enumeration())
        p_init, q_init = get_best_equilibrium(equilibria)

        # Store initial strategy
        self.initial_strategy = p_init[0] if is_player1 else q_init[0]
        self.prior_handler = NashEquilibriumPrior(self.initial_strategy)

        # Initialize Q-values based on Nash equilibrium
        if is_player1:
            payoff_matrix = payoff_matrices[0]
            expected_up = (
                payoff_matrix[0, 0] * q_init[0] + payoff_matrix[0, 1] * q_init[1]
            )
            expected_down = (
                payoff_matrix[1, 0] * q_init[0] + payoff_matrix[1, 1] * q_init[1]
            )
            self.Q_values = {"Up": expected_up, "Down": expected_down}
        else:
            payoff_matrix = payoff_matrices[1]
            expected_left = (
                payoff_matrix[0, 0] * p_init[0] + payoff_matrix[1, 0] * p_init[1]
            )
            expected_right = (
                payoff_matrix[0, 1] * p_init[0] + payoff_matrix[1, 1] * p_init[1]
            )
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
