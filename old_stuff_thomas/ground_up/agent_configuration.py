"""
Fixed version of your agent_configuration.py with all issues corrected.
Save this as agent_configuration.py in your directory.
"""

from typing import Dict, Any
from interfaces import AgentConfig, AgentState
from agent_state import InterpersonalAgentState


class BaseAgentConfig(AgentConfig):
    """Base configuration for agents in the interpersonal simulation"""

    def __init__(
        self,
        # SAC hyperparameters
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_temperature: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        target_entropy: float = None,
        buffer_size: int = 100000,
        batch_size: int = 256,
        hidden_size: int = 256,  # Added this missing parameter
        # Agent personality parameters
        memory_length: int = 50,
        initial_trust: float = 0.0,
        initial_satisfaction: float = 0.0,  # Added this missing parameter
        # Training parameters
        noise_scale: float = 0.1,
    ):
        """
        Initialize agent configuration.

        Args:
            lr_actor: Learning rate for actor network
            lr_critic: Learning rate for critic networks
            lr_temperature: Learning rate for temperature parameter
            gamma: Discount factor
            tau: Soft update coefficient
            alpha: Initial temperature parameter
            target_entropy: Target entropy (None for auto)
            buffer_size: Replay buffer size
            batch_size: Training batch size
            hidden_size: Hidden layer size for networks
            memory_length: How many past actions to remember
            initial_trust: Starting trust level
            initial_satisfaction: Starting satisfaction level
            noise_scale: Exploration noise scale
        """
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_temperature = lr_temperature
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.target_entropy = target_entropy if target_entropy is not None else -1.0
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.memory_length = memory_length
        self.initial_trust = initial_trust
        self.initial_satisfaction = initial_satisfaction
        self.noise_scale = noise_scale

    def get_sac_params(self) -> Dict[str, Any]:
        """Return the parameters for the Soft Actor-Critic algorithm"""
        return {
            "lr_actor": self.lr_actor,
            "lr_critic": self.lr_critic,
            "lr_temperature": self.lr_temperature,
            "gamma": self.gamma,
            "tau": self.tau,
            "alpha": self.alpha,
            "target_entropy": self.target_entropy,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "hidden_size": self.hidden_size,
            "noise_scale": self.noise_scale,
        }

    def get_memory_length(self) -> int:
        """Return the length of the memory buffer for experience replay"""
        return self.memory_length

    def create_initial_state(self) -> AgentState:
        """Create and return an initial state for the agent"""
        return InterpersonalAgentState(
            memory_length=self.memory_length,
            initial_trust=self.initial_trust,
            initial_satisfaction=self.initial_satisfaction,
        )


class CooperativeAgentConfig(BaseAgentConfig):
    """Configuration for cooperative agents"""

    def __init__(self, **kwargs):
        # Cooperative agents start with higher trust and satisfaction
        # They also learn more slowly to reflect their stable nature
        defaults = {
            "lr_actor": 1e-4,
            "lr_critic": 1e-4,
            "gamma": 0.95,  # less forward looking
            "alpha": 0.1,
            "initial_trust": 0.5,  # Higher initial trust
            "initial_satisfaction": 0.3,  # Higher initial satisfaction
            "memory_length": 30,  # Shorter memory for more forgiving nature
            "noise_scale": 0.05,  # Less exploration noise
        }
        defaults.update(kwargs)  # Fixed this line - was missing .update(kwargs)
        super().__init__(**defaults)


class CompetitiveAgentConfig(BaseAgentConfig):
    """Configuration for competitive agents"""

    def __init__(self, **kwargs):
        # Competitive agents start with lower trust
        # They learn faster and explore more
        defaults = {
            "lr_actor": 5e-4,
            "lr_critic": 5e-4,
            "gamma": 0.99,  # More forward-looking
            "alpha": 0.3,  # More exploration
            "initial_trust": -0.3,
            "initial_satisfaction": -0.1,
            "memory_length": 70,  # Longer memory, hold grudges
            "noise_scale": 0.15,  # More noise
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


class AdaptiveAgentConfig(BaseAgentConfig):
    """Configuration for adaptive agents that adjust quickly."""

    def __init__(self, **kwargs):
        # Adaptive agents learn quickly and adjust their behavior
        defaults = {
            "lr_actor": 6e-4,
            "lr_critic": 6e-4,
            "gamma": 0.97,
            "alpha": 0.25,
            "tau": 0.01,  # Faster target updates
            "initial_trust": 0.0,  # Neutral starting point
            "initial_satisfaction": 0.0,
            "memory_length": 50,
            "noise_scale": 0.1,
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


class CautiousAgentConfig(BaseAgentConfig):
    """Configuration for cautious agents that change slowly."""

    def __init__(self, **kwargs):
        # Cautious agents learn slowly and are conservative
        defaults = {
            "lr_actor": 1e-4,
            "lr_critic": 1e-4,
            "gamma": 0.99,
            "alpha": 0.05,  # Very little exploration
            "tau": 0.001,  # Very slow target updates
            "initial_trust": 0.0,
            "initial_satisfaction": 0.0,
            "memory_length": 100,  # Very long memory
            "noise_scale": 0.02,  # Very little noise
        }
        defaults.update(kwargs)
        super().__init__(**defaults)
