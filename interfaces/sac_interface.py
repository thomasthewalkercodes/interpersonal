"""
Abstract interfaces for the interpersonal agent simulation system.

This module defines the core interfaces that all components must implement
to ensure consistency and modularity across the simulation framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np


class AgentState(ABC):
    """
    Abstract base class for agent state representation.

    This defines the interface for storing and managing agent internal state,
    including memory, trust levels, and other psychological variables.
    """

    @abstractmethod
    def update(self, action: float, other_action: float, reward: float) -> None:
        """
        Update the agent's internal state based on interaction outcomes.

        Args:
            action: The action this agent took
            other_action: The action the other agent took
            reward: The reward received for this interaction
        """
        pass

    @abstractmethod
    def get_state_vector(self) -> np.ndarray:
        """
        Get the current state as a vector for neural network input.

        Returns:
            numpy array representing the current state
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the agent state to initial conditions."""
        pass

    @abstractmethod
    def get_trust_level(self) -> float:
        """
        Get the current trust level towards the other agent.

        Returns:
            Trust level (typically in range [-1, 1])
        """
        pass


class AgentConfig(ABC):
    """
    Abstract base class for agent configuration.

    This defines the interface for configuring agent parameters,
    including learning rates, personality traits, and behavioral tendencies.
    """

    @abstractmethod
    def get_sac_params(self) -> Dict[str, Any]:
        """
        Get the SAC (Soft Actor-Critic) algorithm parameters.

        Returns:
            Dictionary containing SAC hyperparameters
        """
        pass

    @abstractmethod
    def get_memory_length(self) -> int:
        """
        Get the length of the agent's memory buffer.

        Returns:
            Number of past interactions to remember
        """
        pass

    @abstractmethod
    def create_initial_state(self) -> AgentState:
        """
        Create and return an initial state for the agent.

        Returns:
            AgentState instance with initial values
        """
        pass


class ReinforcementLearner(ABC):
    """
    Abstract base class for reinforcement learning agents.

    This defines the core interface that all RL agents must implement
    for training and action selection in the simulation environment.
    """

    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = True) -> float:
        """
        Select an action based on the current state.

        Args:
            state: Current state vector
            training: Whether agent is in training mode (affects exploration)

        Returns:
            Selected action (warmth level in [-1, 1])
        """
        pass

    @abstractmethod
    def store_transition(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store a transition in the agent's experience buffer.

        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode ended
        """
        pass

    @abstractmethod
    def train_step(self) -> Optional[Dict[str, float]]:
        """
        Perform one training step using stored experiences.

        Returns:
            Dictionary of training metrics (losses, etc.) or None if not enough data
        """
        pass

    @abstractmethod
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to file.

        Args:
            filepath: Path where to save the model
        """
        pass

    @abstractmethod
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from file.

        Args:
            filepath: Path to the saved model
        """
        pass


class Environment(ABC):
    """
    Abstract base class for simulation environments.

    This defines the interface for environments where agents interact,
    following the standard RL environment pattern.
    """

    @abstractmethod
    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reset the environment to initial state.

        Returns:
            Tuple of initial states for (agent1, agent2)
        """
        pass

    @abstractmethod
    def step(
        self, action1: float, action2: float
    ) -> Tuple[np.ndarray, np.ndarray, float, float, bool]:
        """
        Execute one step of interaction between two agents.

        Args:
            action1: Action from first agent
            action2: Action from second agent

        Returns:
            Tuple of (next_state1, next_state2, reward1, reward2, done)
        """
        pass

    @abstractmethod
    def get_state_dim(self) -> int:
        """
        Get the dimensionality of the state space.

        Returns:
            Number of dimensions in state vector
        """
        pass


class PayoffCalculator(ABC):
    """
    Abstract base class for payoff/reward calculation functions.

    This defines the interface for computing rewards based on agent interactions,
    allowing for different types of interpersonal dynamics.
    """

    @abstractmethod
    def calculate_payoff(
        self, action1: float, action2: float, context: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, float]:
        """
        Calculate payoffs for both agents based on their actions.

        Args:
            action1: Action taken by first agent
            action2: Action taken by second agent
            context: Optional context information (state, history, etc.)

        Returns:
            Tuple of (reward1, reward2)
        """
        pass

    @abstractmethod
    def get_optimal_response(self, other_action: float) -> float:
        """
        Get the theoretically optimal response to another agent's action.

        Args:
            other_action: The other agent's action

        Returns:
            Optimal response action
        """
        pass


class Trainer(ABC):
    """
    Abstract base class for training multiple agents.

    This defines the interface for managing the training process
    of multiple interacting agents.
    """

    @abstractmethod
    def train(self, save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Train the agents through interaction.

        Args:
            save_dir: Optional directory to save models during training

        Returns:
            Dictionary containing training results and metrics
        """
        pass

    @abstractmethod
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the trained agents without further learning.

        Args:
            num_episodes: Number of episodes to run for evaluation

        Returns:
            Dictionary containing evaluation metrics
        """
        pass


class MetricsCollector(ABC):
    """
    Abstract base class for collecting and analyzing simulation metrics.

    This defines the interface for tracking various aspects of agent
    behavior and learning throughout the simulation.
    """

    @abstractmethod
    def record_episode(
        self,
        episode: int,
        agent1_reward: float,
        agent2_reward: float,
        episode_length: int,
        additional_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record metrics for a completed episode.

        Args:
            episode: Episode number
            agent1_reward: Total reward for agent 1
            agent2_reward: Total reward for agent 2
            episode_length: Number of steps in the episode
            additional_metrics: Any additional metrics to record
        """
        pass

    @abstractmethod
    def record_step(
        self,
        step: int,
        action1: float,
        action2: float,
        reward1: float,
        reward2: float,
        state1: np.ndarray,
        state2: np.ndarray,
    ) -> None:
        """
        Record metrics for a single interaction step.

        Args:
            step: Global step number
            action1: Action taken by agent 1
            action2: Action taken by agent 2
            reward1: Reward received by agent 1
            reward2: Reward received by agent 2
            state1: State of agent 1
            state2: State of agent 2
        """
        pass

    @abstractmethod
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics for the collected data.

        Returns:
            Dictionary containing summary statistics
        """
        pass

    @abstractmethod
    def save_metrics(self, filepath: str) -> None:
        """
        Save collected metrics to file.

        Args:
            filepath: Path where to save the metrics
        """
        pass


class ExperimentManager(ABC):
    """
    Abstract base class for managing experiments and studies.

    This defines the interface for running systematic experiments
    with different agent configurations and conditions.
    """

    @abstractmethod
    def setup_experiment(self, experiment_config: Dict[str, Any]) -> None:
        """
        Set up an experiment with the given configuration.

        Args:
            experiment_config: Dictionary defining the experiment parameters
        """
        pass

    @abstractmethod
    def run_experiment(self) -> Dict[str, Any]:
        """
        Run the configured experiment.

        Returns:
            Dictionary containing experiment results
        """
        pass

    @abstractmethod
    def analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze experimental results and generate insights.

        Args:
            results: Raw experimental results

        Returns:
            Dictionary containing analysis and insights
        """
        pass


# Type hints for common data structures
StateVector = np.ndarray
Action = float
Reward = float
AgentID = str
EpisodeResults = Dict[str, Any]
TrainingMetrics = Dict[str, float]
SimulationResults = Dict[str, Any]
