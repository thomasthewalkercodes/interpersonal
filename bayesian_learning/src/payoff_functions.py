# src/payoff_functions.py
"""
Gaussian payoff functions for continuous interpersonal dynamics
Based on your specification: warm-warm matching gets most points,
falls off gradually toward cold-cold, mismatching hurts most
"""

import numpy as np
from typing import Union


def gaussian_matching_payoff(
    my_action: Union[float, np.ndarray],
    opponent_action: Union[float, np.ndarray],
    peak_payoff: float = 10.0,
    mismatch_penalty: float = -5.0,
    matching_bonus: float = 3.0,
    falloff_rate: float = 2.0,
) -> float:
    """
    Gaussian payoff function emphasizing warm-warm matching.

    Payoff structure:
    - Highest payoff at warm-warm (1.0, 1.0)
    - Gradual falloff toward cold-cold (0.0, 0.0)
    - Steep penalty for mismatching (warm-cold, cold-warm)

    Args:
        my_action: My warmth level [0, 1]
        opponent_action: Opponent's warmth level [0, 1]
        peak_payoff: Maximum payoff at perfect warm matching
        mismatch_penalty: Base penalty for mismatching
        matching_bonus: Bonus for any level of matching
        falloff_rate: How steep the falloff is (higher = steeper)

    Returns:
        Payoff value
    """

    # Ensure scalar values
    my_warmth = float(my_action) if np.isscalar(my_action) else my_action[0]
    opp_warmth = (
        float(opponent_action) if np.isscalar(opponent_action) else opponent_action[0]
    )

    # Clip to valid range
    my_warmth = np.clip(my_warmth, 0.0, 1.0)
    opp_warmth = np.clip(opp_warmth, 0.0, 1.0)

    # Calculate matching vs mismatching
    difference = abs(my_warmth - opp_warmth)
    average_warmth = (my_warmth + opp_warmth) / 2.0

    # Gaussian centered on perfect matching (difference = 0)
    # with peak at warm end (high average_warmth)

    # Matching component: Gaussian peaked at difference = 0
    matching_component = np.exp(-falloff_rate * difference**2)

    # Warmth bonus: Linear bonus for higher average warmth
    # This makes warm-warm better than cold-cold
    warmth_component = average_warmth * matching_bonus

    # Mismatch penalty: Additional penalty for large differences
    mismatch_component = mismatch_penalty * (difference > 0.3) * difference

    # Combine components
    payoff = peak_payoff * matching_component + warmth_component + mismatch_component

    return payoff


def interpersonal_gaussian_payoff(
    my_action: Union[float, np.ndarray],
    opponent_action: Union[float, np.ndarray],
    my_ideal_warmth: float = 0.8,
    ideal_opponent_warmth: float = 0.7,
    tolerance: float = 0.2,
) -> float:
    """
    Gaussian payoff centered on my ideal interpersonal configuration.

    Args:
        my_action: My warmth level [0, 1]
        opponent_action: Opponent's warmth level [0, 1]
        my_ideal_warmth: My preferred warmth level
        ideal_opponent_warmth: My preferred opponent warmth level
        tolerance: How tolerant I am of deviations (higher = more tolerant)

    Returns:
        Payoff value
    """

    my_warmth = float(my_action) if np.isscalar(my_action) else my_action[0]
    opp_warmth = (
        float(opponent_action) if np.isscalar(opponent_action) else opponent_action[0]
    )

    # Clip to valid range
    my_warmth = np.clip(my_warmth, 0.0, 1.0)
    opp_warmth = np.clip(opp_warmth, 0.0, 1.0)

    # Distance from ideal configuration
    my_error = (my_warmth - my_ideal_warmth) ** 2
    opp_error = (opp_warmth - ideal_opponent_warmth) ** 2

    # Gaussian payoff centered on ideal point
    total_error = my_error + opp_error
    payoff = 10 * np.exp(-total_error / (2 * tolerance**2))

    # Baseline adjustment to avoid always-negative payoffs
    payoff = payoff - 2

    return payoff


def symmetric_gaussian_payoff(
    my_action: Union[float, np.ndarray],
    opponent_action: Union[float, np.ndarray],
    cooperation_peak: float = 0.8,
    peak_payoff: float = 8.0,
    defection_valley: float = 0.2,
    valley_payoff: float = 1.0,
    mismatch_penalty: float = -3.0,
    sharpness: float = 3.0,
) -> float:
    """
    Symmetric Gaussian payoff function matching your description:
    - Peak at warm-warm matching
    - Gradual falloff to cold-cold
    - Strong penalty for mismatching

    Args:
        my_action: My warmth level [0, 1]
        opponent_action: Opponent's warmth level [0, 1]
        cooperation_peak: Warmth level where cooperation peaks
        peak_payoff: Payoff at peak cooperation
        defection_valley: Warmth level of "cold-cold" region
        valley_payoff: Payoff at cold-cold matching
        mismatch_penalty: Additional penalty for mismatching
        sharpness: How sharp the matching preference is

    Returns:
        Payoff value
    """

    my_warmth = float(my_action) if np.isscalar(my_action) else my_action[0]
    opp_warmth = (
        float(opponent_action) if np.isscalar(opponent_action) else opponent_action[0]
    )

    # Clip to valid range
    my_warmth = np.clip(my_warmth, 0.0, 1.0)
    opp_warmth = np.clip(opp_warmth, 0.0, 1.0)

    # Calculate the "matching quality"
    difference = abs(my_warmth - opp_warmth)
    average_warmth = (my_warmth + opp_warmth) / 2.0

    # Base payoff based on average warmth level
    # Linear interpolation between valley and peak
    base_payoff = valley_payoff + (peak_payoff - valley_payoff) * (
        average_warmth - defection_valley
    ) / (cooperation_peak - defection_valley)

    # Matching bonus/penalty using Gaussian
    matching_multiplier = np.exp(-sharpness * difference**2)

    # Apply mismatch penalty for large differences
    mismatch_component = mismatch_penalty * (1 - matching_multiplier)

    # Final payoff
    payoff = base_payoff * matching_multiplier + mismatch_component

    return payoff


def create_personality_payoff_function(personality_type: str = "secure"):
    """
    Creates a payoff function based on personality type.

    Args:
        personality_type: "secure", "anxious", "avoidant"

    Returns:
        Payoff function configured for that personality
    """

    if personality_type == "secure":
        # Secure: Likes warm matching, tolerant of some mismatch
        def secure_payoff(my_action, opp_action):
            return gaussian_matching_payoff(
                my_action,
                opp_action,
                peak_payoff=10.0,
                mismatch_penalty=-2.0,  # Lower penalty
                matching_bonus=3.0,
                falloff_rate=1.5,  # Gentler falloff
            )

        return secure_payoff

    elif personality_type == "anxious":
        # Anxious: Strong preference for warm matching, harsh mismatch penalty
        def anxious_payoff(my_action, opp_action):
            return gaussian_matching_payoff(
                my_action,
                opp_action,
                peak_payoff=12.0,
                mismatch_penalty=-8.0,  # Harsh penalty
                matching_bonus=4.0,
                falloff_rate=3.0,  # Sharp falloff
            )

        return anxious_payoff

    elif personality_type == "avoidant":
        # Avoidant: Prefers moderate warmth, less penalty for cold matching
        def avoidant_payoff(my_action, opp_action):
            return interpersonal_gaussian_payoff(
                my_action,
                opp_action,
                my_ideal_warmth=0.4,  # Prefers lower warmth
                ideal_opponent_warmth=0.4,
                tolerance=0.3,  # More tolerant
            )

        return avoidant_payoff

    else:
        # Default to symmetric
        return lambda my_action, opp_action: symmetric_gaussian_payoff(
            my_action, opp_action
        )


# Example usage and testing functions
def test_payoff_functions():
    """Test the payoff functions to ensure they work as expected"""

    print("Testing Gaussian Payoff Functions")
    print("=" * 40)

    # Test points
    test_points = [
        (1.0, 1.0, "Warm-Warm (perfect match)"),
        (0.0, 0.0, "Cold-Cold (low match)"),
        (1.0, 0.0, "Warm-Cold (mismatch)"),
        (0.0, 1.0, "Cold-Warm (mismatch)"),
        (0.5, 0.5, "Medium-Medium (medium match)"),
        (0.8, 0.7, "High-High (close match)"),
    ]

    for my_action, opp_action, description in test_points:
        payoff1 = gaussian_matching_payoff(my_action, opp_action)
        payoff2 = symmetric_gaussian_payoff(my_action, opp_action)

        print(
            f"{description:25} | Matching: {payoff1:6.2f} | Symmetric: {payoff2:6.2f}"
        )

    print("\nTesting Personality-Based Payoffs")
    print("=" * 40)

    personalities = ["secure", "anxious", "avoidant"]

    for personality in personalities:
        payoff_func = create_personality_payoff_function(personality)
        warm_warm = payoff_func(0.9, 0.9)
        mismatch = payoff_func(0.9, 0.1)

        print(
            f"{personality:10} | Warm-Warm: {warm_warm:6.2f} | Mismatch: {mismatch:6.2f}"
        )


if __name__ == "__main__":
    test_payoff_functions()
