"""
Core functions for the Interpersonal Circumplex Model
"""

import numpy as np
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


class CircumplexSpace:
    """
    Handles conversions and operations in the interpersonal circumplex space.

    The circumplex uses:
    - Angle (0-360°): Interpersonal style
        - 0°/360°: Warm
        - 90°: Dominant
        - 180°: Cold
        - 270°: Submissive
    - Radius (0-1): Intensity of behavior
    """

    @staticmethod
    def cartesian_to_circumplex(warmth: float, dominance: float) -> Tuple[float, float]:
        """Convert Cartesian coordinates to polar (angle, radius)."""
        angle = np.degrees(np.arctan2(warmth, dominance)) % 360
        radius = np.sqrt(warmth**2 + dominance**2)
        radius = min(radius, 1.0)  # Cap at unit circle
        return angle, radius

    @staticmethod
    def circumplex_to_cartesian(angle: float, radius: float) -> Tuple[float, float]:
        """Convert polar coordinates to Cartesian (warmth, dominance)."""
        angle_rad = np.radians(angle)
        dominance = radius * np.cos(angle_rad)
        warmth = radius * np.sin(angle_rad)
        return warmth, dominance

    @staticmethod
    def compute_optimal_match(angle: float) -> float:
        """
        Compute the optimal matching angle.
        Same warmth + mirrored dominance means rotating 180° around the warmth axis.
        """
        # Mirror across the vertical (warmth) axis
        if angle <= 180:
            return 180 - angle
        else:
            return 540 - angle

    @staticmethod
    def angular_distance(angle1: float, angle2: float) -> float:
        """Compute the shortest angular distance between two angles."""
        diff = abs(angle1 - angle2)
        return min(diff, 360 - diff)

    @staticmethod
    def behavioral_distance(
        angle1: float, radius1: float, angle2: float, radius2: float
    ) -> float:
        """
        Compute the distance between two behaviors in circumplex space.
        Uses Euclidean distance in the transformed Cartesian space.
        """
        w1, d1 = CircumplexSpace.circumplex_to_cartesian(angle1, radius1)
        w2, d2 = CircumplexSpace.circumplex_to_cartesian(angle2, radius2)
        return np.sqrt((w1 - w2) ** 2 + (d1 - d2) ** 2)


class PayoffMatrix:
    """
    Calculates payoffs based on behavioral matching in the circumplex.
    """

    def __init__(
        self,
        max_payoff: float = 10.0,
        angle_weight: float = 1,
        radius_weight: float = 1,
    ):
        """
        Initialize payoff calculator.

        Args:
            max_payoff: Maximum possible payoff for perfect match
            angle_weight: Weight for angular matching (style compatibility)
            radius_weight: Weight for radius matching (intensity compatibility)
        """
        self.max_payoff = max_payoff
        self.angle_weight = angle_weight
        self.radius_weight = radius_weight

    def calculate_payoff(
        self, behavior1: Tuple[float, float], behavior2: Tuple[float, float]
    ) -> float:
        """
        Calculate payoff for agent1 based on behavior matching.

        Perfect match: Same warmth level + mirrored dominance
        """
        angle1, radius1 = behavior1
        angle2, radius2 = behavior2

        # Calculate optimal angle for agent1 given agent2's behavior
        optimal_angle = CircumplexSpace.compute_optimal_match(angle2)

        # Angular similarity (0 to 1, where 1 is perfect match)
        angle_diff = CircumplexSpace.angular_distance(angle1, optimal_angle)
        angle_similarity = 1 - (angle_diff / 180)  # Normalize to [0, 1]

        # Radius similarity (0 to 1)
        radius_similarity = 1 - abs(radius1 - radius2)

        # Combined payoff
        payoff = self.max_payoff * (
            self.angle_weight * angle_similarity
            + self.radius_weight * radius_similarity
        )

        return payoff


class BayesianBelief:
    """
    Represents an agent's Bayesian belief about another agent's behavior.
    Uses a 2D Gaussian distribution over the circumplex space.
    """

    def __init__(self, initial_mean: np.ndarray = None, initial_cov: np.ndarray = None):
        """
        Initialize belief distribution.

        Args:
            initial_mean: [warmth, dominance] mean vector
            initial_cov: 2x2 covariance matrix
        """
        if initial_mean is None:
            initial_mean = np.array([0.0, 0.0])  # Neutral prior
        if initial_cov is None:
            # Start with high uncertainty
            initial_cov = np.array([[0.5, 0.0], [0.0, 0.5]])

        self.mean = initial_mean
        self.cov = initial_cov
        self.distribution = multivariate_normal(mean=self.mean, cov=self.cov)

    def update(
        self, observed_behavior: Tuple[float, float], learning_rate: float = 0.1
    ):
        """
        Update belief based on observed behavior using Bayesian updating.

        Args:
            observed_behavior: (angle, radius) of observed behavior
            learning_rate: How quickly to update beliefs (0-1)
        """
        # Convert observed behavior to Cartesian
        warmth, dominance = CircumplexSpace.circumplex_to_cartesian(*observed_behavior)
        observation = np.array([warmth, dominance])

        # Bayesian update (simplified using exponential moving average)
        self.mean = (1 - learning_rate) * self.mean + learning_rate * observation

        # Update covariance (reduce uncertainty over time)
        observation_diff = observation - self.mean
        obs_cov = np.outer(observation_diff, observation_diff)
        self.cov = (1 - learning_rate * 0.5) * self.cov + learning_rate * 0.5 * obs_cov

        # Ensure covariance remains positive definite
        self.cov = self.cov + np.eye(2) * 0.01

        # Update distribution
        self.distribution = multivariate_normal(mean=self.mean, cov=self.cov)

    def sample_beliefs(self, n_samples: int = 100) -> np.ndarray:
        """
        Sample from the belief distribution.

        Returns:
            Array of (angle, radius) samples
        """
        samples_cartesian = self.distribution.rvs(size=n_samples)
        if samples_cartesian.ndim == 1:
            samples_cartesian = samples_cartesian.reshape(1, -1)

        samples_polar = []
        for sample in samples_cartesian:
            angle, radius = CircumplexSpace.cartesian_to_circumplex(
                sample[0], sample[1]
            )
            samples_polar.append([angle, radius])

        return np.array(samples_polar)

    def get_expected_behavior(self) -> Tuple[float, float]:
        """Get the expected (mean) behavior from beliefs."""
        return CircumplexSpace.cartesian_to_circumplex(self.mean[0], self.mean[1])

    def get_uncertainty(self) -> float:
        """Get a measure of uncertainty (determinant of covariance)."""
        return np.sqrt(np.linalg.det(self.cov))
