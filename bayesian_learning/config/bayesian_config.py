# config/bayesian_config.py
"""
Configuration parameters for Bayesian interpersonal dynamics simulation
"""

import numpy as np
from typing import Dict, Any


def create_default_config() -> Dict[str, Any]:
    """
    Creates default configuration parameters for Bayesian simulation
    that matches your existing interpersonal dynamics model structure

    Returns:
        Dictionary containing configuration parameters
    """

    config = {
        # Simulation parameters
        "simulation": {
            "n_rounds": 300,  # Number of interaction rounds (from your model)
            "n_replications": 50,  # Number of simulation runs for statistics
            "action_selection_method": "thompson",  # "thompson", "ucb", "softmax"
            "verbose": False,
            "random_seed": 42,
        },
        # Agent parameters
        "agents": {
            # Prior beliefs about opponent (Beta distribution parameters)
            "alpha_prior": 1.0,  # Prior count for warm actions
            "beta_prior": 1.0,  # Prior count for cold actions
            # Psychological parameters (matching your existing model)
            "lambda_loss": 1.0,  # Loss aversion parameter
            "temperature": 1.0,  # Exploration temperature for softmax
        },
        # Payoff matrices (your interpersonal dynamics structure)
        "payoffs": {
            # Default cooperative game (both benefit from mutual warmth)
            "default_matrix": np.array(
                [[1, 0], [3, 2]]  # Cold-Cold, Cold-Warm  # Warm-Cold, Warm-Warm
            ),
            # Prisoner's dilemma variant
            "prisoners_dilemma": np.array(
                [[1, 0], [3, 2]]  # Cold-Cold, Cold-Warm  # Warm-Cold, Warm-Warm
            ),
            # Trust game variant (from your model)
            "trust_game": np.array(
                [[2, 1], [4, 3]]  # Cold-Cold, Cold-Warm  # Warm-Cold, Warm-Warm
            ),
        },
    }

    return config


def create_agent_preset(agent_type: str = "secure") -> Dict[str, Any]:
    """
    Creates preset configurations for different personality types
    based on your existing character models (Ruth, Mei, etc.)

    Args:
        agent_type: Type of agent ("secure", "anxious", "avoidant", "neutral")

    Returns:
        Dictionary containing agent-specific parameters

    Raises:
        ValueError: If agent_type is not recognized
    """

    presets = {
        # Secure attachment (Mei-like)
        "secure": {
            "alpha_prior": 2.0,  # Optimistic about others' warmth
            "beta_prior": 1.0,  # Less expectation of coldness
            "lambda_loss": 1.0,  # Normal loss aversion
            "temperature": 0.8,  # Moderate exploration
        },
        # Anxious attachment (Ruth-like)
        "anxious": {
            "alpha_prior": 1.0,  # Uncertain about others
            "beta_prior": 2.0,  # Expecting more coldness
            "lambda_loss": 2.0,  # High loss aversion (from your model)
            "temperature": 1.2,  # Higher exploration due to uncertainty
        },
        # Avoidant attachment
        "avoidant": {
            "alpha_prior": 0.5,  # Low expectations of warmth
            "beta_prior": 2.0,  # Strong expectation of coldness
            "lambda_loss": 1.5,  # Moderate-high loss aversion
            "temperature": 0.5,  # Low exploration, stick to safe strategies
        },
        # Balanced/neutral agent
        "neutral": {
            "alpha_prior": 1.0,  # No prior bias
            "beta_prior": 1.0,  # No prior bias
            "lambda_loss": 1.0,  # Normal loss aversion
            "temperature": 1.0,  # Standard exploration
        },
        # Highly optimistic (for testing)
        "optimistic": {
            "alpha_prior": 3.0,  # Very optimistic about warmth
            "beta_prior": 1.0,  # Low expectation of coldness
            "lambda_loss": 0.8,  # Low loss aversion
            "temperature": 0.6,  # Low exploration (confident)
        },
        # Highly pessimistic (for testing)
        "pessimistic": {
            "alpha_prior": 0.5,  # Very low expectation of warmth
            "beta_prior": 3.0,  # High expectation of coldness
            "lambda_loss": 2.5,  # Very high loss aversion
            "temperature": 1.5,  # High exploration (uncertain)
        },
    }

    if agent_type not in presets:
        raise ValueError(f"agent_type must be one of: {list(presets.keys())}")

    return presets[agent_type]


def create_payoff_matrix(matrix_type: str = "default") -> np.ndarray:
    """
    Creates payoff matrices for different interaction scenarios

    Args:
        matrix_type: Type of payoff matrix ("default", "prisoners_dilemma",
                    "trust_game", "competitive", "pure_coordination")

    Returns:
        2x2 numpy array representing the payoff matrix

    Raises:
        ValueError: If matrix_type is not recognized
    """

    matrices = {
        "default": np.array(
            [[1, 0], [3, 2]]  # Cold-Cold, Cold-Warm  # Warm-Cold, Warm-Warm
        ),
        "prisoners_dilemma": np.array(
            [
                [1, 0],  # Cold-Cold, Cold-Warm (temptation to defect)
                [3, 2],  # Warm-Cold, Warm-Warm
            ]
        ),
        "trust_game": np.array(
            [
                [2, 1],  # Cold-Cold, Cold-Warm (safer but lower payoff)
                [4, 3],  # Warm-Cold, Warm-Warm (risky but higher payoff)
            ]
        ),
        "competitive": np.array(
            [
                [0, -1],  # Cold-Cold, Cold-Warm (zero-sum like)
                [1, 0],  # Warm-Cold, Warm-Warm
            ]
        ),
        "pure_coordination": np.array(
            [
                [2, 0],  # Cold-Cold, Cold-Warm (coordination game)
                [0, 2],  # Warm-Cold, Warm-Warm
            ]
        ),
        # Asymmetric game (for testing different roles)
        "asymmetric": np.array(
            [[1, 2], [0, 3]]  # Cold-Cold, Cold-Warm  # Warm-Cold, Warm-Warm
        ),
    }

    if matrix_type not in matrices:
        raise ValueError(f"matrix_type must be one of: {list(matrices.keys())}")

    return matrices[matrix_type]


def load_config_from_dict(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Loads configuration from a dictionary, filling in defaults for missing values

    Args:
        config_dict: Dictionary containing partial or complete configuration

    Returns:
        Complete configuration dictionary with defaults filled in
    """

    default_config = create_default_config()

    # Deep merge config_dict into default_config
    def merge_dicts(default: Dict, custom: Dict) -> Dict:
        result = default.copy()
        for key, value in custom.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = merge_dicts(result[key], value)
            else:
                result[key] = value
        return result

    return merge_dicts(default_config, config_dict)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validates configuration parameters

    Args:
        config: Configuration dictionary to validate

    Returns:
        True if configuration is valid

    Raises:
        ValueError: If configuration contains invalid values
    """

    # Check simulation parameters
    sim_config = config.get("simulation", {})

    if sim_config.get("n_rounds", 0) <= 0:
        raise ValueError("n_rounds must be positive")

    if sim_config.get("n_replications", 0) <= 0:
        raise ValueError("n_replications must be positive")

    valid_methods = ["thompson", "ucb", "softmax"]
    if sim_config.get("action_selection_method") not in valid_methods:
        raise ValueError(f"action_selection_method must be one of {valid_methods}")

    # Check agent parameters
    agent_config = config.get("agents", {})

    if agent_config.get("alpha_prior", 0) <= 0:
        raise ValueError("alpha_prior must be positive")

    if agent_config.get("beta_prior", 0) <= 0:
        raise ValueError("beta_prior must be positive")

    if agent_config.get("lambda_loss", 0) <= 0:
        raise ValueError("lambda_loss must be positive")

    if agent_config.get("temperature", 0) <= 0:
        raise ValueError("temperature must be positive")

    # Check payoff matrices
    payoff_config = config.get("payoffs", {})

    for matrix_name, matrix in payoff_config.items():
        if not isinstance(matrix, np.ndarray):
            raise ValueError(f"Payoff matrix '{matrix_name}' must be numpy array")

        if matrix.shape != (2, 2):
            raise ValueError(f"Payoff matrix '{matrix_name}' must be 2x2")

    return True
