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
    prior_weight: float  # Weight for Nash equilibrium prior (0-1)
