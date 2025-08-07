"""
Personality configurations for different agent types.

Each personality defines how an agent perceives, learns, and adapts in social interactions.
"""

PERSONALITIES = {
    "balanced": {
        "description": "Well-adjusted individual with neutral biases",
        "base_behavior": (45, 0.6),  # Warm-dominant, moderate intensity
        "warmth_bias": 0.0,
        "dominance_bias": 0.0,
        "learning_rate": 0.15,
        "adaptation_rate": 0.1,
        "mood_sensitivity": 0.3,
        "risk_aversion": 0.5,
        "initial_mood": 0.2,
    },
    "rejection_sensitive": {
        "description": "Highly sensitive to rejection, perceives others as colder",
        "base_behavior": (225, 0.4),  # Cold-submissive, low intensity
        "warmth_bias": -0.3,  # Sees others as colder than they are
        "dominance_bias": 0.1,  # Slightly overestimates dominance
        "learning_rate": 0.25,  # Quick to update beliefs (hypervigilant)
        "adaptation_rate": 0.2,  # Rapidly changes behavior
        "mood_sensitivity": 0.8,  # Mood highly affected by interactions
        "risk_aversion": 0.8,  # Avoids risky social behaviors
        "initial_mood": -0.2,
    },
    "depressed": {
        "description": "Low mood, withdrawn, perceives negativity",
        "base_behavior": (270, 0.3),  # Submissive, very low intensity
        "warmth_bias": -0.2,  # Sees less warmth in others
        "dominance_bias": 0.2,  # Others seem more dominant
        "learning_rate": 0.05,  # Slow to update beliefs (learned helplessness)
        "adaptation_rate": 0.05,  # Resistant to change
        "mood_sensitivity": 0.2,  # Mood changes slowly
        "risk_aversion": 0.9,  # Very risk-averse
        "initial_mood": -0.7,
    },
    "narcissistic": {
        "description": "Grandiose, dominant, lacks empathy",
        "base_behavior": (90, 0.9),  # Highly dominant, high intensity
        "warmth_bias": -0.4,  # Dismissive of others' warmth
        "dominance_bias": -0.3,  # Underestimates others' dominance
        "learning_rate": 0.05,  # Slow to learn about others (self-focused)
        "adaptation_rate": 0.03,  # Very resistant to changing behavior
        "mood_sensitivity": 0.4,  # Moderate mood sensitivity (ego protection)
        "risk_aversion": 0.2,  # Risk-seeking in social dominance
        "initial_mood": 0.5,
    },
    "anxious": {
        "description": "Socially anxious, uncertain, seeks approval",
        "base_behavior": (200, 0.5),  # Warm-submissive, moderate intensity
        "warmth_bias": 0.1,  # Slight positive bias (hopeful)
        "dominance_bias": 0.3,  # Overestimates others' dominance
        "learning_rate": 0.3,  # Very quick to update (hypervigilant)
        "adaptation_rate": 0.25,  # Frequently adjusts behavior
        "mood_sensitivity": 0.7,  # High mood reactivity
        "risk_aversion": 0.7,  # Risk-averse
        "initial_mood": -0.3,
    },
    "aggressive": {
        "description": "Hostile, confrontational, sees threats",
        "base_behavior": (350, 0.8),  # Cold-dominant, high intensity
        "warmth_bias": -0.5,  # Sees hostility in others
        "dominance_bias": 0.0,  # Accurate dominance perception
        "learning_rate": 0.1,  # Moderate learning
        "adaptation_rate": 0.15,  # Moderate adaptation
        "mood_sensitivity": 0.6,  # Reactive mood
        "risk_aversion": 0.3,  # Risk-seeking in confrontation
        "initial_mood": -0.4,
    },
    "people_pleaser": {
        "description": "Overly accommodating, seeks harmony",
        "base_behavior": (180, 0.7),  # Pure warm, high intensity
        "warmth_bias": 0.3,  # Sees others as warmer (optimistic)
        "dominance_bias": 0.0,  # Accurate dominance perception
        "learning_rate": 0.2,  # Moderate-fast learning
        "adaptation_rate": 0.3,  # Quick to accommodate
        "mood_sensitivity": 0.5,  # Moderate mood sensitivity
        "risk_aversion": 0.6,  # Somewhat risk-averse
        "initial_mood": 0.1,
    },
    "avoidant": {
        "description": "Emotionally distant, self-reliant",
        "base_behavior": (315, 0.4),  # Cold-neutral, low intensity
        "warmth_bias": -0.2,  # Slight negative bias
        "dominance_bias": 0.0,  # Accurate perception
        "learning_rate": 0.08,  # Slow to learn (disengaged)
        "adaptation_rate": 0.05,  # Very slow to change
        "mood_sensitivity": 0.1,  # Low mood reactivity (emotional numbing)
        "risk_aversion": 0.8,  # Avoids emotional risks
        "initial_mood": 0.0,
    },
    "manic": {
        "description": "Elevated mood, impulsive, energetic",
        "base_behavior": (60, 0.95),  # Warm-dominant, maximum intensity
        "warmth_bias": 0.4,  # Sees everyone as friendly
        "dominance_bias": -0.2,  # Underestimates others' dominance
        "learning_rate": 0.35,  # Rapid but unstable learning
        "adaptation_rate": 0.4,  # Very quick changes
        "mood_sensitivity": 0.9,  # Extreme mood swings
        "risk_aversion": 0.1,  # Very risk-seeking
        "initial_mood": 0.8,
    },
    "paranoid": {
        "description": "Suspicious, distrustful, defensive",
        "base_behavior": (330, 0.6),  # Cold-dominant, moderate intensity
        "warmth_bias": -0.6,  # Sees hostility everywhere
        "dominance_bias": 0.4,  # Sees others as threatening (dominant)
        "learning_rate": 0.02,  # Extremely slow to trust
        "adaptation_rate": 0.1,  # Slow to change (rigid)
        "mood_sensitivity": 0.5,  # Moderate mood sensitivity
        "risk_aversion": 0.95,  # Extremely risk-averse
        "initial_mood": -0.5,
    },
}


def get_personality(name: str) -> dict:
    """
    Get a personality configuration by name.

    Args:
        name: Name of the personality type

    Returns:
        Dictionary of personality parameters

    Raises:
        ValueError: If personality name not found
    """
    if name not in PERSONALITIES:
        available = ", ".join(PERSONALITIES.keys())
        raise ValueError(f"Personality '{name}' not found. Available: {available}")

    return PERSONALITIES[name].copy()


def list_personalities() -> list:
    """Get list of available personality types."""
    return list(PERSONALITIES.keys())


def describe_personality(name: str) -> str:
    """Get description of a personality type."""
    if name not in PERSONALITIES:
        return f"Personality '{name}' not found"

    p = PERSONALITIES[name]
    return f"{name}: {p['description']}"


def create_custom_personality(base_personality: str = "balanced", **overrides) -> dict:
    """
    Create a custom personality based on an existing one with overrides.

    Args:
        base_personality: Name of base personality to modify
        **overrides: Parameters to override

    Returns:
        Modified personality dictionary
    """
    personality = get_personality(base_personality)
    personality.update(overrides)
    return personality
