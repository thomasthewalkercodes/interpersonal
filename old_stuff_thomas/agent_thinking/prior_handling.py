from abc import ABC, abstractmethod
from dataclasses import dataclass
from Configurations import QLearningConfig


@dataclass
class PriorConfig:
    """Configuration for prior influence"""

    weight: float  # Weight of prior (0-1)


class PriorHandler(ABC):
    """Abstract base class for handling different types of priors"""

    @abstractmethod
    def blend_probabilities(
        self, learned_prob: float, config: QLearningConfig
    ) -> float:
        pass


class NashEquilibriumPrior(PriorHandler):
    """Handles Nash equilibrium prior probabilities"""

    def __init__(self, nash_prob: float):
        self.nash_prob = nash_prob

    def blend_probabilities(
        self, learned_prob: float, config: QLearningConfig
    ) -> float:
        """Blend learned probabilities with Nash equilibrium prior"""
        return (
            1 - config.prior_weight
        ) * learned_prob + config.prior_weight * self.nash_prob
