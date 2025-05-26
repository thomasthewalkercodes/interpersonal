import numpy as np
import nashpy as nash  # Changed from 'import nash' to 'import nashpy as nash'
from typing import Dict, Tuple, List
from dataclasses import dataclass
from Configurations import QLearningConfig, A1, A2
from .prior_handling import NashEquilibriumPrior


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
        """Choose action using softmax policy with numerical stability"""
        # Add small constant to avoid overflow
        max_q = max(self.Q_values.values())
        exp_q = {
            action: np.exp(self.config.beta * (q_val - max_q))
            for action, q_val in self.Q_values.items()
        }
        total_exp_q = sum(exp_q.values())

        # Avoid division by zero
        if total_exp_q > 0:
            probs = {action: eq / total_exp_q for action, eq in exp_q.items()}
        else:
            # If numerical issues occur, use uniform distribution
            probs = {action: 1.0 / len(self.Q_values) for action in self.Q_values}

        # Choose action based on probabilities
        actions = list(probs.keys())
        probabilities = list(probs.values())
        return np.random.choice(actions, p=probabilities)

    def update(self, action: str, payoff: float) -> None:
        """Update Q-values with numerical stability checks"""
        old_value = self.Q_values[action]
        # Clip payoff to reasonable range to avoid overflow
        payoff = np.clip(payoff, -1e6, 1e6)

        # Calculate value with prospect theory
        if payoff >= self.reference_point:
            value = (payoff - self.reference_point) ** self.config.rho
        else:
            value = (
                -self.config.lambda_val
                * (self.reference_point - payoff) ** self.config.rho
            )

        # Clip the update to avoid overflow
        delta = np.clip(value - old_value, -1e6, 1e6)
        self.Q_values[action] = old_value + self.config.alpha * delta

        # Update reference point using EMA
        self.reference_point = (
            1 - self.config.ema_weight
        ) * self.reference_point + self.config.ema_weight * payoff


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


class ContinuousQLearningAgent:
    """Q-learning agent for continuous circular action space"""

    def __init__(self, config: QLearningConfig, is_player1: bool):
        self.config = config
        self.is_player1 = is_player1
        self.reference_point = 0.0

        # Initialize action parameters (x, y coordinates)
        self.current_x = np.random.uniform(-0.5, 0.5)
        self.current_y = np.random.uniform(-0.5, 0.5)

        # Initialize simple function approximator weights
        self.weights_x = np.random.normal(0, 0.1, 4)  # weights for x coordinate
        self.weights_y = np.random.normal(0, 0.1, 4)  # weights for y coordinate

        # Learning history
        self.last_action = (self.current_x, self.current_y)
        self.action_history = []

    def _feature_vector(self, x: float, y: float) -> np.ndarray:
        """Compute feature vector for function approximation"""
        return np.array([1.0, x, y, x * y])

    def _predict_value(self, x: float, y: float, weights: np.ndarray) -> float:
        """Predict value for a coordinate using function approximation"""
        features = self._feature_vector(x, y)
        return np.dot(features, weights)

    def choose_action(self) -> Tuple[float, float]:
        """Choose next action using continuous action space"""
        # Add exploration noise
        noise_x = np.random.normal(0, 0.1 / self.config.beta)
        noise_y = np.random.normal(0, 0.1 / self.config.beta)

        # Update position with noise and ensure within unit circle
        new_x = self.current_x + noise_x
        new_y = self.current_y + noise_y

        # Normalize if outside unit circle
        dist = np.sqrt(new_x**2 + new_y**2)
        if dist > 1:
            new_x /= dist
            new_y /= dist

        self.last_action = (new_x, new_y)
        self.action_history.append(self.last_action)
        return new_x, new_y

    def update(self, payoff: float) -> None:
        """Update agent's policy based on received payoff"""
        # Calculate value using prospect theory
        if payoff >= self.reference_point:
            value = (payoff - self.reference_point) ** self.config.rho
        else:
            value = (
                -self.config.lambda_val
                * (self.reference_point - payoff) ** self.config.rho
            )

        # Update function approximator weights
        x, y = self.last_action
        features = self._feature_vector(x, y)

        # Update weights for both coordinates
        td_error_x = value - self._predict_value(x, y, self.weights_x)
        td_error_y = value - self._predict_value(x, y, self.weights_y)

        self.weights_x += self.config.alpha * td_error_x * features
        self.weights_y += self.config.alpha * td_error_y * features

        # Update reference point using exponential moving average
        self.reference_point = (
            self.config.ema_weight * payoff
            + (1 - self.config.ema_weight) * self.reference_point
        )

        # Update current position
        self.current_x, self.current_y = self.last_action


class GameEnvironment:
    def __init__(self, game_config, agent1, agent2):
        self.game_config = game_config
        self.agent1 = agent1
        self.agent2 = agent2
        if game_config.game_type == "matrix":
            self.A1 = game_config.matrix_config["A1"]
            self.A2 = game_config.matrix_config["A2"]
        else:
            self.circular_config = game_config.circular_config

    def step(self):
        """Execute one game step"""
        if self.game_config.game_type == "matrix":
            # Matrix game logic
            action1 = self.agent1.choose_action()
            action2 = self.agent2.choose_action()

            idx1 = 0 if action1 == "Up" else 1
            idx2 = 0 if action2 == "Left" else 1

            payoff1 = self.A1[idx1, idx2]
            payoff2 = self.A2[idx1, idx2]

            self.agent1.update(payoff1)
            self.agent2.update(payoff2)

            return action1, action2, payoff1, payoff2
        else:
            # Circular game logic
            action1 = self.agent1.choose_action()
            action2 = self.agent2.choose_action()

            # Calculate payoffs using circular game rules
            dist1 = np.sqrt(action1[0] ** 2 + action1[1] ** 2)
            dist2 = np.sqrt(action2[0] ** 2 + action2[1] ** 2)

            if dist1 > 1 or dist2 > 1:  # Invalid moves outside circle
                payoff1, payoff2 = 0, 0
            else:
                # Calculate similarities
                communion_similarity = np.exp(
                    -self.circular_config.w_c * (action2[0] - action1[0]) ** 2
                )
                agency_similarity = np.exp(
                    -self.circular_config.w_a * (action2[1] + action1[1]) ** 2
                )

                payoff = self.circular_config.max_payoff * (
                    communion_similarity * agency_similarity
                )
                payoff1 = payoff2 = payoff  # Same payoff for both agents for now

            self.agent1.update(payoff1)
            self.agent2.update(payoff2)

            return action1, action2, payoff1, payoff2
