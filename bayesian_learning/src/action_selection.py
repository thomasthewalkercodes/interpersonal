# src/action_selection.py
"""
Bayesian action selection for interpersonal dynamics
"""

import numpy as np
from typing import Literal
from .bayesian_agent import BayesianAgent


def select_action_thompson(agent: BayesianAgent, payoff_matrix: np.ndarray) -> int:
    """
    Selects action using Thompson sampling from posterior beliefs.
    This naturally balances exploitation and exploration.

    Args:
        agent: BayesianAgent object
        payoff_matrix: 2x2 numpy array, payoff structure [own_action, opponent_action]

    Returns:
        Selected action (0=Cold, 1=Warm)

    Raises:
        TypeError: If inputs have wrong type
        ValueError: If payoff_matrix has wrong shape
    """

    if not isinstance(agent, BayesianAgent):
        raise TypeError("agent must be a BayesianAgent object")

    if not isinstance(payoff_matrix, np.ndarray) or payoff_matrix.shape != (2, 2):
        raise ValueError("payoff_matrix must be a 2x2 numpy array")

    # Sample opponent's warm probability from posterior Beta distribution
    sampled_opponent_prob = np.random.beta(agent.belief_alpha, agent.belief_beta)

    # Calculate expected payoffs for each action given sampled probability
    # Action 0 (Cold): E[payoff] = prob_warm * payoff[0,1] + (1-prob_warm) * payoff[0,0]
    # Action 1 (Warm): E[payoff] = prob_warm * payoff[1,1] + (1-prob_warm) * payoff[1,0]

    expected_payoff_cold = (
        sampled_opponent_prob * payoff_matrix[0, 1]
        + (1 - sampled_opponent_prob) * payoff_matrix[0, 0]
    )

    expected_payoff_warm = (
        sampled_opponent_prob * payoff_matrix[1, 1]
        + (1 - sampled_opponent_prob) * payoff_matrix[1, 0]
    )

    # Apply loss aversion (prospect theory component from your model)
    if expected_payoff_cold < 0:
        expected_payoff_cold *= agent.lambda_loss

    if expected_payoff_warm < 0:
        expected_payoff_warm *= agent.lambda_loss

    # Select action with higher expected payoff
    selected_action = 1 if expected_payoff_warm > expected_payoff_cold else 0

    return selected_action


def select_action_ucb(
    agent: BayesianAgent, payoff_matrix: np.ndarray, confidence_factor: float = 1.96
) -> int:
    """
    Alternative action selection using Upper Confidence Bound (UCB)
    with Bayesian confidence intervals.

    Args:
        agent: BayesianAgent object
        payoff_matrix: 2x2 numpy array, payoff structure
        confidence_factor: Confidence scaling factor (default: 1.96 for 95% CI)

    Returns:
        Selected action (0=Cold, 1=Warm)
    """

    if not isinstance(agent, BayesianAgent):
        raise TypeError("agent must be a BayesianAgent object")

    # Get belief statistics
    belief_stats = agent.get_belief_stats()

    # Calculate optimistic estimate (mean + confidence interval)
    optimistic_opponent_prob = min(
        1.0, belief_stats["mean"] + confidence_factor * belief_stats["std_dev"]
    )

    # Calculate expected payoffs using optimistic estimate
    expected_payoff_cold = (
        optimistic_opponent_prob * payoff_matrix[0, 1]
        + (1 - optimistic_opponent_prob) * payoff_matrix[0, 0]
    )

    expected_payoff_warm = (
        optimistic_opponent_prob * payoff_matrix[1, 1]
        + (1 - optimistic_opponent_prob) * payoff_matrix[1, 0]
    )

    # Apply loss aversion
    if expected_payoff_cold < 0:
        expected_payoff_cold *= agent.lambda_loss

    if expected_payoff_warm < 0:
        expected_payoff_warm *= agent.lambda_loss

    # Select action with higher UCB
    selected_action = 1 if expected_payoff_warm > expected_payoff_cold else 0

    return selected_action


def select_action_softmax(agent: BayesianAgent, payoff_matrix: np.ndarray) -> int:
    """
    Probabilistic action selection using softmax over expected payoffs.

    Args:
        agent: BayesianAgent object
        payoff_matrix: 2x2 numpy array, payoff structure

    Returns:
        Selected action (0=Cold, 1=Warm)
    """

    if not isinstance(agent, BayesianAgent):
        raise TypeError("agent must be a BayesianAgent object")

    # Calculate expected payoffs using current belief mean
    prob_warm = agent.opponent_warm_prob

    expected_payoff_cold = (
        prob_warm * payoff_matrix[0, 1] + (1 - prob_warm) * payoff_matrix[0, 0]
    )

    expected_payoff_warm = (
        prob_warm * payoff_matrix[1, 1] + (1 - prob_warm) * payoff_matrix[1, 0]
    )

    # Apply loss aversion
    if expected_payoff_cold < 0:
        expected_payoff_cold *= agent.lambda_loss

    if expected_payoff_warm < 0:
        expected_payoff_warm *= agent.lambda_loss

    # Softmax probabilities
    payoffs = np.array([expected_payoff_cold, expected_payoff_warm])
    exp_payoffs = np.exp(payoffs / agent.temperature)
    action_probs = exp_payoffs / np.sum(exp_payoffs)

    # Sample action according to probabilities
    selected_action = np.random.choice([0, 1], p=action_probs)

    return selected_action


def select_action(
    agent: BayesianAgent,
    payoff_matrix: np.ndarray,
    method: Literal["thompson", "ucb", "softmax"] = "thompson",
    **kwargs,
) -> int:
    """
    Unified action selection interface that dispatches to specific methods.

    Args:
        agent: BayesianAgent object
        payoff_matrix: 2x2 numpy array, payoff structure
        method: Action selection method ("thompson", "ucb", "softmax")
        **kwargs: Additional arguments passed to specific selection methods

    Returns:
        Selected action (0=Cold, 1=Warm)

    Raises:
        ValueError: If method is not recognized
    """

    if method == "thompson":
        return select_action_thompson(agent, payoff_matrix)
    elif method == "ucb":
        return select_action_ucb(agent, payoff_matrix, **kwargs)
    elif method == "softmax":
        return select_action_softmax(agent, payoff_matrix)
    else:
        raise ValueError(
            f"Unknown action selection method: {method}. "
            "Must be 'thompson', 'ucb', or 'softmax'"
        )
