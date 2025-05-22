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
    prior_weight=0.2,
)

config2 = QLearningConfig(
    alpha=0.2,
    beta=0.8,
    gamma=0.88,
    rho=0,
    lambda_val=1,
    ema_weight=0.1,
    prior_weight=0.2,
)


@dataclass
class TestConfiguration:
    """Configuration for multiple test runs"""

    n_repetitions: int = 20  # Number of times to repeat each configuration
    n_rounds: int = 1000  # Number of rounds per repetition
    variable_ranges: Dict[str, List[float]] = None  # Values for each variable to test
    save_individual_data: bool = True  # Whether to save round-by-round data
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
                "prior_weight": 0.2,
            }

        if self.variable_ranges is None:
            # Default ranges for testing different parameters
            self.variable_ranges = {
                "alpha": [0.1],
                "beta": [1.0],
            }

    def generate_configs(self) -> List[Dict]:
        """Generate all combinations of specified variables"""
        from itertools import product

        # Get all combinations of values for the specified variables
        var_names = list(self.variable_ranges.keys())
        var_values = list(self.variable_ranges.values())
        combinations = list(product(*var_values))

        configs = []
        for combo in combinations:
            # Start with base configuration
            config = self.base_config.copy()
            # Update only the variables we're testing
            for var_name, value in zip(var_names, combo):
                config[var_name] = value
            configs.append(config)

        return configs


# Define default test configuration
test_config = TestConfiguration(
    n_repetitions=5,
    n_rounds=1000,
    variable_ranges={
        "alpha": [0.1, 0.3, 0.8],
        "beta": [1.0, 2, 3.0],
    },
    base_config={
        "alpha": 0.2,
        "beta": 2.0,
        "gamma": 0.9,
        "rho": 0.0,
        "lambda_val": 2.25,
        "ema_weight": 0.1,
        "prior_weight": 0.2,
    },
)
