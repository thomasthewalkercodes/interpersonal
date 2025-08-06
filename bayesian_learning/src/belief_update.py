# src/belief_update.py
"""
Bayesian belief updating functions for interpersonal dynamics
"""

import numpy as np
from typing import Union
from .bayesian_agent import BayesianAgent


def update_beliefs(agent: BayesianAgent, opponent_action: int) -> BayesianAgent:
    """
    Updates agent's beliefs about opponent using Bayesian posterior updating.
    Uses Beta-Binomial conjugate prior for efficient updating.

    Args:
        agent: BayesianAgent object to update
        opponent_action: Opponent's action (0=Cold, 1=Warm)

    Returns:
        Updated BayesianAgent object

    Raises:
        ValueError: If opponent_action is not 0 or 1
        TypeError: If agent is not a BayesianAgent
    """

    # Validate inputs
    if not isinstance(agent, BayesianAgent):
        raise TypeError("agent must be a BayesianAgent object")

    if opponent_action not in [0, 1]:
        raise ValueError("opponent_action must be 0 (Cold) or 1 (Warm)")

    # Bayesian update using Beta-Binomial conjugacy
    # If opponent_action = 1 (Warm): increment alpha
    # If opponent_action = 0 (Cold): increment beta

    if opponent_action == 1:
        agent.belief_alpha += 1
    else:
        agent.belief_beta += 1

    # Update point estimate of opponent's warmth probability
    # This is the posterior mean of Beta distribution
    agent.opponent_warm_prob = agent.belief_alpha / (
        agent.belief_alpha + agent.belief_beta
    )

    # Add to opponent history
    agent.opponent_history.append(opponent_action)

    return agent


def calculate_belief_uncertainty(agent: BayesianAgent) -> float:
    """
    Calculates the uncertainty in agent's beliefs about opponent.
    Uses variance of Beta distribution as uncertainty measure.

    Args:
        agent: BayesianAgent object

    Returns:
        Belief uncertainty (variance of Beta distribution)

    Raises:
        TypeError: If agent is not a BayesianAgent
    """

    if not isinstance(agent, BayesianAgent):
        raise TypeError("agent must be a BayesianAgent object")

    # Variance of Beta(alpha, beta) = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))
    alpha = agent.belief_alpha
    beta = agent.belief_beta

    variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))

    return variance


def reset_agent_beliefs(
    agent: BayesianAgent, alpha_prior: float = 1.0, beta_prior: float = 1.0
) -> BayesianAgent:
    """
    Resets agent's beliefs to specified priors while keeping other parameters.
    Useful for running multiple simulations with the same agent.

    Args:
        agent: BayesianAgent object to reset
        alpha_prior: New prior count for warm actions
        beta_prior: New prior count for cold actions

    Returns:
        Agent with reset beliefs but same other parameters
    """

    if not isinstance(agent, BayesianAgent):
        raise TypeError("agent must be a BayesianAgent object")

    # Reset belief parameters
    agent.belief_alpha = alpha_prior
    agent.belief_beta = beta_prior

    # Recalculate opponent warm probability
    agent.opponent_warm_prob = alpha_prior / (alpha_prior + beta_prior)

    # Clear history
    agent.action_history.clear()
    agent.opponent_history.clear()
    agent.payoff_history.clear()

    return agent


def get_belief_distribution_samples(
    agent: BayesianAgent, n_samples: int = 1000
) -> np.ndarray:
    """
    Generates samples from the agent's current belief distribution.
    Useful for visualization and Monte Carlo analysis.

    Args:
        agent: BayesianAgent object
        n_samples: Number of samples to generate

    Returns:
        NumPy array of samples from Beta(alpha, beta) distribution
    """

    if not isinstance(agent, BayesianAgent):
        raise TypeError("agent must be a BayesianAgent object")

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    return np.random.beta(agent.belief_alpha, agent.belief_beta, size=n_samples)
