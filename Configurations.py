import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from agent_thinking.Circular_game_environment import CircularGameConfig

A1 = np.array([[3, 1], [1, 2]])  # Player 1's payoff matrix
A2 = np.array([[3, 1], [1, 2]])  # Player 2's payoff matrix
N_ROUNDS = 1000


@dataclass
class CircularAgentConfig:
    """Configuration for a single agent in circular game"""

    w_c: float = 1.0  # communion weight
    w_a: float = 1.0  # agency weight
    max_payoff: float = 10.0


@dataclass
class CircularGameConfig:
    """Configuration for circular interpersonal game"""

    agent1_config: CircularAgentConfig = None
    agent2_config: CircularAgentConfig = None

    def __post_init__(self):
        if self.agent1_config is None:
            self.agent1_config = CircularAgentConfig()
        if self.agent2_config is None:
            self.agent2_config = CircularAgentConfig(w_c=0.5, w_a=2.0)


@dataclass
class GameConfig:
    """Game type configuration"""

    game_type: str = "circular"  # "matrix" or "circular"
    matrix_config: Dict = None
    circular_config: CircularGameConfig = None

    def __post_init__(self):
        if self.game_type == "matrix":
            if self.matrix_config is None:
                self.matrix_config = {
                    "A1": A1,
                    "A2": A2,
                }
        elif self.game_type == "circular":
            if self.circular_config is None:
                self.circular_config = CircularGameConfig()


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
    beta=3,
    gamma=0.88,
    rho=0.2,
    lambda_val=2,
    ema_weight=0.1,
    prior_weight=0,
)

config2 = QLearningConfig(
    alpha=0.3,
    beta=3,  # Different beta for agent 2
    gamma=0.88,
    rho=0.2,
    lambda_val=2,
    ema_weight=0.1,
    prior_weight=0,
)


@dataclass
class VisualizationConfig:
    """Configuration for visualization options"""

    show_heatmap: bool = True
    show_agent1_heatmap: bool = True  # Toggle between agent 1 and 2 heatmap
    update_heatmap: bool = True  # Whether to update heatmap during animation
    animation_interval: int = 20  # ms between frames
    trail_length: int = 20  # Number of previous positions to show


# Update TestConfiguration to include visualization settings
@dataclass
class TestConfiguration:
    """Configuration for multiple test runs"""

    n_repetitions: int = 20
    n_rounds: int = 1000
    variable_ranges: Dict[str, List[float]] = None
    base_config: Dict[str, float] = None
    game_config: GameConfig = None  # Changed from game_config to GameConfig
    viz_config: VisualizationConfig = None

    def __post_init__(self):
        if self.base_config is None:
            self.base_config = {
                "alpha": 0.2,
                "beta": 2.0,
                "gamma": 0.9,
                "rho": 0.0,
                "lambda_val": 2.25,
                "ema_weight": 0.1,
                "prior_weight": 0,
            }
        if self.game_config is None:
            self.game_config = (
                GameConfig()
            )  # Changed from game_config() to GameConfig()
        if self.viz_config is None:
            self.viz_config = VisualizationConfig()


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
        "beta": 3.0,
        "gamma": 0.9,
        "rho": 0.0,
        "lambda_val": 2.25,
        "ema_weight": 0.1,
        "prior_weight": 0,
    },
    game_config=GameConfig(
        game_type="circular",
        circular_config=CircularGameConfig(
            agent1_config=CircularAgentConfig(
                w_c=1,  #  1 Agent 1 prefers strong communion matching
                w_a=1,  # 0.5 But cares less about agency mirroring
                max_payoff=10.0,
            ),
            agent2_config=CircularAgentConfig(
                w_c=1,  # 0.5 Agent 2 cares less about communion matching
                w_a=1,  # 2 But strongly prefers agency mirroring
                max_payoff=10.0,
            ),
        ),
    ),
)
