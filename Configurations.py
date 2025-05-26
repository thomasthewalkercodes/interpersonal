import numpy as np
from dataclasses import dataclass
from typing import List, Dict


A1 = np.array([[3, 1], [1, 2]])  # Player 1's payoff matrix
A2 = np.array([[3, 1], [1, 2]])  # Player 2's payoff matrix
N_ROUNDS = 1000


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
    beta=0.8,
    gamma=0.88,
    rho=0,
    lambda_val=1,
    ema_weight=0.1,
    prior_weight=0,
)

config2 = QLearningConfig(
    alpha=0.2,
    beta=0.8,
    gamma=0.88,
    rho=0,
    lambda_val=1,
    ema_weight=0.1,
    prior_weight=0,
)


@dataclass
class TestConfiguration:
    """Configuration for multiple test runs"""

    n_repetitions: int = 20  # Number of times to repeat each configuration
    n_rounds: int = 1000  # Number of rounds per repetition
    variable_ranges: Dict[str, List[float]] = None  # Values for each variable to test
    base_config: Dict[str, float] = None  # Base configuration for non-tested parameters

    def __post_init__(self):
        if self.base_config is None:
            # Default base configuration
            self.base_config = {
                "alpha": 0.2,
                "beta": 2.0,
                "gamma": 0.9,
                "rho": 0.0,
                "lambda_val": 2.25,
                "ema_weight": 0.1,
                "prior_weight": 0,
            }


# Define default test configuration
test_config = TestConfiguration(
    n_repetitions=5,
    n_rounds=1000,
    variable_ranges={
        "beta": [1.0, 2.0, 3.0],
        "lambda_val": [1.0, 2.25, 4.0],
    },
    base_config={
        "alpha": 0.2,
        "beta": 2.0,
        "gamma": 0.9,
        "rho": 0.0,
        "lambda_val": 2.25,
        "ema_weight": 0.1,
        "prior_weight": 0,
    },
)
