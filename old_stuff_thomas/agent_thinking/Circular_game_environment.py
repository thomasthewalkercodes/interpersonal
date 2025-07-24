import numpy as np
from dataclasses import dataclass


@dataclass
class CircularGameConfig:
    """Configuration for circular interpersonal game"""

    w_c: float = 1.0  # communion weight
    w_a: float = 1.0  # agency weight
    max_payoff: float = 10.0
    radius: float = 1.0  # unit circle radius


class CircularGameEnvironment:
    def __init__(self, config: CircularGameConfig):
        self.config = config
        self.agent1_pos = None
        self.agent2_pos = None

    def calculate_payoff(self, pos1, pos2):
        """Calculate payoff for given positions"""
        x1, y1 = pos1
        x2, y2 = pos2

        # Normalize positions if outside unit circle
        dist1 = np.sqrt(x1**2 + y1**2)
        dist2 = np.sqrt(x2**2 + y2**2)
        norm1 = max(1, dist1)
        norm2 = max(1, dist2)

        x1_norm, y1_norm = x1 / norm1, y1 / norm1
        x2_norm, y2_norm = x2 / norm2, y2 / norm2

        # Calculate similarities
        communion_similarity = np.exp(-self.config.w_c * (x2_norm - x1_norm) ** 2)
        agency_similarity = np.exp(-self.config.w_a * (y2_norm + y1_norm) ** 2)

        return self.config.max_payoff * (communion_similarity * agency_similarity)

    def step(self, action1, action2):
        """Execute one game step with continuous actions"""
        self.agent1_pos = action1
        self.agent2_pos = action2

        if (
            np.sqrt(action1[0] ** 2 + action1[1] ** 2) > self.config.radius
            or np.sqrt(action2[0] ** 2 + action2[1] ** 2) > self.config.radius
        ):
            return None  # Invalid moves outside circle

        payoff1 = self.calculate_payoff(action1, action2)
        payoff2 = self.calculate_payoff(action2, action1)  # Same calculation for now

        return action1, action2, payoff1, payoff2
