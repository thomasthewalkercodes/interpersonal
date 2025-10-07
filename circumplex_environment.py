"Core functions for the Interpersonal Circumplex Model"

import numpy as np
from typing import Tuple
from scipy.stats import multivariate_normal


class CircumplexSpace:
    """
    This class handles conversions and operation sin the interpersonal circumplex space."
    warm = 0 or 360
    dominant 90
    colld 180
    submissive 270
    Radius = 0-1: intensity of behavior
    Also the sinus wave has to be described in here and translated but that will take some time
    """

    @staticmethod
    def cartesian_to_circumplex(warmth: float, dominance: float) -> Tuple[float, float]:
        """
        Converts Cartesian coordinates to polar (angle, radius)
        # Assumption:
        - capped radius to 1.0
        - angle is random if both coordinates are zero (could be set to zero instead)
        """
        angle = (
            np.degrees(np.arctan2(dominance, warmth)) % 360
        )  # arctan2(y,x) so be careful with the order, %360 normalizes the angle to [0, 360)
        radius = np.hypot(
            warmth, dominance
        )  # hypot better than sqrt(warmth^2 + dominance^2) (high numbers)
        radius = min(radius, 1.0)
        if warmth == 0 and dominance == 0:
            radius = 0
            angle = np.random(0, 361)  # its 361 because upper bound is exclusionary
        return angle, radius

    @staticmethod
    def circumplex_to_cartesian(angle: float, radius: float) -> Tuple[float, float]:
        """
        Converts polar coordinates (angle, radius) to Cartesian coordinates (warmth, dominance)
        """
        angle_rad = np.radians(angle)
        dominance = radius * np.sin(angle_rad)
        warmth = radius * np.cos(angle_rad)
        return warmth, dominance

    @staticmethod
    def compute_optimal_match(angle: float) -> float:
        return 360 - angle

    @staticmethod
    def compute_opposite_match(angle: float) -> float:
        """Returns the opposite angle in the circumplex space."""
        return (angle + 180) % 360

    @staticmethod
    def angular_distance(angle1: float, angle2: float) -> float:
        """If angle1 = 10 and angle2 = 350, diff = 340, but the shortest distance is 360 - 340 = 20.
        The function returns 20."""
        diff = abs(angle1 - angle2)
        return min(diff, 360 - diff)

    @staticmethod
    def behavioral_distance(
        angle1: float, radius1: float, angle2: float, radius2: float
    ) -> float:
        """Compute distance from angle and radius of two points in the circumplex space."""
        w1, d1 = CircumplexSpace.circumplex_to_cartesian(angle1, radius1)
        w2, d2 = CircumplexSpace.circumplex_to_cartesian(angle2, radius2)
        return np.sqrt((w1 - w2) ** 2 + (d1 - d2) ** 2)


class PayoffMatrix:
    """
    Calculates payoffs
    Assumption:
    - angle weight and radius weight are both 1.0 by default
    - max_payoff is 10.0 by default
    """

    def __init__(
        self,
        max_payoff: float = 10.0,
        angle_weight: float = 1,  # style compatibility
        radius_weight: float = 1,  # intensity compatibility
    ):
        self.max_payoff = max_payoff
        self.angle_weight = angle_weight
        self.radius_weight = radius_weight

    def calculate_payoff(
        self, behavior1: Tuple[float, float], behavior2: Tuple[float, float]
    ) -> float:
        """
        Perfect match: same warmth level + mirrored dominance
        """
        angle1, radius1 = behavior1
        angle2, radius2 = behavior2

        optimal_angle = CircumplexSpace.compute_optimal_match(angle2)

        angle_diff = CircumplexSpace.angular_distance(angle1, optimal_angle)
        angle_similarity = 1 - (angle_diff / 180)

        radius_similarity = 1 - abs(radius1 - radius2)

        payoff = self.max_payoff * (
            self.angle_weight * angle_similarity
            + self.radius_weight * radius_similarity
        )

        return payoff
