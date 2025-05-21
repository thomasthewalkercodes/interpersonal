import numpy as np
from dataclasses import dataclass


A1 = np.array([[5, 2], [4, 1]])  # Player 1's payoff matrix
A2 = np.array([[5, 4], [0, 1]])  # Player 2's payoff matrix

N_ROUNDS = 200


@dataclass
class QLearningConfig:
    """Configuration parameters for Q-learning agent"""

    alpha: float  # Learning rate
    beta: float  # Temperature parameter for softmax
    gamma: float  # Discount factor
    rho: float  # Risk aversion parameter
    lambda_val: float  # Loss aversion parameter
    ema_weight: float  # Weight for exponential moving average
    prior_weight: float  # Weight for Nash equilibrium prior (0-1)


# Default configurations for agents
config1 = QLearningConfig(
    alpha=0.2,
    beta=4.0,
    gamma=0.88,
    rho=0,
    lambda_val=1,
    ema_weight=0.1,
    prior_weight=0.2,
)

config2 = QLearningConfig(
    alpha=0.2,
    beta=4.0,
    gamma=0.88,
    rho=0,
    lambda_val=1,
    ema_weight=0.11,
    prior_weight=0.2,
)
