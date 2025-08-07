# src/continuous_action_selection.py
"""
Action selection methods for continuous Bayesian agents
"""

import numpy as np
from scipy.optimize import minimize_scalar, minimize
from typing import Callable, Union, Tuple

# Change this relative import to absolute import
from continuous_bayesian_agent import ContinuousBayesianAgent


def select_action_thompson_continuous(
    agent: ContinuousBayesianAgent,
    payoff_function: Callable[[float, float], float],
    n_opponent_samples: int = 20,
    optimization_bounds: Tuple[float, float] = (0.0, 1.0),
) -> float:
    """
    Thompson sampling for continuous action selection.

    Args:
        agent: ContinuousBayesianAgent object
        payoff_function: Function that takes (my_action, opponent_action) -> payoff
        n_opponent_samples: Number of opponent action samples for expectation
        optimization_bounds: Bounds for my action optimization

    Returns:
        Selected continuous action (scalar for 1D warmth)
    """

    def expected_payoff(my_action):
        """Expected payoff given current beliefs about opponent"""

        # Sample opponent actions from current belief distribution
        opponent_samples = agent.predict_opponent_action(n_samples=n_opponent_samples)

        payoffs = []
        for opp_sample in opponent_samples:
            opp_action = (
                opp_sample[0] if isinstance(opp_sample, np.ndarray) else opp_sample
            )

            # Calculate payoff for this opponent sample
            payoff = payoff_function(my_action, opp_action)

            # Apply loss aversion from psychological model
            if payoff < 0:
                payoff *= agent.lambda_loss

            payoffs.append(payoff)

        return np.mean(payoffs)

    # Optimize my action to maximize expected payoff
    def neg_expected_payoff(my_action):
        return -expected_payoff(my_action)

    # Use scalar optimization for 1D action space
    result = minimize_scalar(
        neg_expected_payoff, bounds=optimization_bounds, method="bounded"
    )

    # Clip to bounds and return
    selected_action = np.clip(result.x, optimization_bounds[0], optimization_bounds[1])
    return selected_action


def select_action_ucb_continuous(
    agent: ContinuousBayesianAgent,
    payoff_function: Callable[[float, float], float],
    confidence_factor: float = 1.96,
    optimization_bounds: Tuple[float, float] = (0.0, 1.0),
) -> float:
    """
    Upper Confidence Bound action selection for continuous actions.

    Args:
        agent: ContinuousBayesianAgent object
        payoff_function: Function that takes (my_action, opponent_action) -> payoff
        confidence_factor: Confidence level for UCB (default: 1.96 for 95%)
        optimization_bounds: Bounds for my action optimization

    Returns:
        Selected continuous action
    """

    # Get current belief statistics
    belief_stats = agent.get_belief_stats()

    # Calculate optimistic opponent action (mean + confidence * std)
    opponent_mean = belief_stats["expected_opponent_mean"][0]
    opponent_var = belief_stats["expected_opponent_cov"][0, 0]
    opponent_std = np.sqrt(max(opponent_var, 1e-6))  # Avoid numerical issues

    # Optimistic estimate
    optimistic_opponent_action = min(
        1.0, opponent_mean + confidence_factor * opponent_std
    )

    def ucb_payoff(my_action):
        """Payoff using optimistic opponent estimate"""
        payoff = payoff_function(my_action, optimistic_opponent_action)

        # Apply loss aversion
        if payoff < 0:
            payoff *= agent.lambda_loss

        return payoff

    # Optimize my action
    def neg_ucb_payoff(my_action):
        return -ucb_payoff(my_action)

    result = minimize_scalar(
        neg_ucb_payoff, bounds=optimization_bounds, method="bounded"
    )

    selected_action = np.clip(result.x, optimization_bounds[0], optimization_bounds[1])
    return selected_action


def select_action_softmax_continuous(
    agent: ContinuousBayesianAgent,
    payoff_function: Callable[[float, float], float],
    n_action_samples: int = 20,
    optimization_bounds: Tuple[float, float] = (0.0, 1.0),
) -> float:
    """
    Softmax action selection for continuous actions.
    Samples candidate actions and selects according to softmax probabilities.

    Args:
        agent: ContinuousBayesianAgent object
        payoff_function: Function that takes (my_action, opponent_action) -> payoff
        n_action_samples: Number of candidate actions to consider
        optimization_bounds: Bounds for action sampling

    Returns:
        Selected continuous action
    """

    # Sample candidate actions uniformly
    candidate_actions = np.random.uniform(
        optimization_bounds[0], optimization_bounds[1], size=n_action_samples
    )

    # Calculate expected payoff for each candidate action
    expected_payoffs = []

    for candidate_action in candidate_actions:
        # Expected payoff calculation (simplified - use mean opponent action)
        belief_stats = agent.get_belief_stats()
        opponent_mean = belief_stats["expected_opponent_mean"][0]

        payoff = payoff_function(candidate_action, opponent_mean)

        # Apply loss aversion
        if payoff < 0:
            payoff *= agent.lambda_loss

        expected_payoffs.append(payoff)

    expected_payoffs = np.array(expected_payoffs)

    # Softmax probabilities
    exp_payoffs = np.exp(expected_payoffs / agent.temperature)
    probabilities = exp_payoffs / np.sum(exp_payoffs)

    # Sample action according to probabilities
    selected_idx = np.random.choice(n_action_samples, p=probabilities)
    selected_action = candidate_actions[selected_idx]

    return selected_action


def select_action_greedy_continuous(
    agent: ContinuousBayesianAgent,
    payoff_function: Callable[[float, float], float],
    optimization_bounds: Tuple[float, float] = (0.0, 1.0),
) -> float:
    """
    Greedy action selection using mean opponent belief.

    Args:
        agent: ContinuousBayesianAgent object
        payoff_function: Function that takes (my_action, opponent_action) -> payoff
        optimization_bounds: Bounds for action optimization

    Returns:
        Selected continuous action
    """

    # Get mean opponent action from beliefs
    belief_stats = agent.get_belief_stats()
    opponent_mean = belief_stats["expected_opponent_mean"][0]

    def greedy_payoff(my_action):
        """Payoff using mean opponent action"""
        payoff = payoff_function(my_action, opponent_mean)

        # Apply loss aversion
        if payoff < 0:
            payoff *= agent.lambda_loss

        return payoff

    # Optimize my action
    def neg_greedy_payoff(my_action):
        return -greedy_payoff(my_action)

    result = minimize_scalar(
        neg_greedy_payoff, bounds=optimization_bounds, method="bounded"
    )

    selected_action = np.clip(result.x, optimization_bounds[0], optimization_bounds[1])
    return selected_action


def select_action_continuous(
    agent: ContinuousBayesianAgent,
    payoff_function: Callable[[float, float], float],
    method: str = "thompson",
    **kwargs,
) -> float:
    """
    Unified interface for continuous action selection.

    Args:
        agent: ContinuousBayesianAgent object
        payoff_function: Function that takes (my_action, opponent_action) -> payoff
        method: Action selection method ("thompson", "ucb", "softmax", "greedy")
        **kwargs: Additional arguments for specific methods

    Returns:
        Selected continuous action

    Raises:
        ValueError: If method is not recognized
    """

    if method == "thompson":
        return select_action_thompson_continuous(agent, payoff_function, **kwargs)
    elif method == "ucb":
        return select_action_ucb_continuous(agent, payoff_function, **kwargs)
    elif method == "softmax":
        return select_action_softmax_continuous(agent, payoff_function, **kwargs)
    elif method == "greedy":
        return select_action_greedy_continuous(agent, payoff_function, **kwargs)
    else:
        raise ValueError(
            f"Unknown action selection method: {method}. "
            "Must be 'thompson', 'ucb', 'softmax', or 'greedy'"
        )


def add_exploration_noise(
    action: float, noise_level: float = 0.05, bounds: Tuple[float, float] = (0.0, 1.0)
) -> float:
    """
    Add Gaussian exploration noise to selected action.

    Args:
        action: Base action to add noise to
        noise_level: Standard deviation of noise
        bounds: Bounds to clip final action

    Returns:
        Action with added noise, clipped to bounds
    """

    noise = np.random.normal(0, noise_level)
    noisy_action = action + noise
    return np.clip(noisy_action, bounds[0], bounds[1])


def select_action_epsilon_greedy_continuous(
    agent: ContinuousBayesianAgent,
    payoff_function: Callable[[float, float], float],
    epsilon: float = 0.1,
    optimization_bounds: Tuple[float, float] = (0.0, 1.0),
) -> float:
    """
    Epsilon-greedy action selection for continuous actions.

    Args:
        agent: ContinuousBayesianAgent object
        payoff_function: Function that takes (my_action, opponent_action) -> payoff
        epsilon: Probability of random exploration
        optimization_bounds: Bounds for actions

    Returns:
        Selected continuous action
    """

    if np.random.random() < epsilon:
        # Explore: random action
        return np.random.uniform(optimization_bounds[0], optimization_bounds[1])
    else:
        # Exploit: greedy action
        return select_action_greedy_continuous(
            agent, payoff_function, optimization_bounds
        )


def select_action_with_personality(
    agent: ContinuousBayesianAgent,
    payoff_function: Callable[[float, float], float],
    personality_bias: float = 0.0,
    bias_strength: float = 0.2,
) -> float:
    """
    Action selection with personality bias.

    Args:
        agent: ContinuousBayesianAgent object
        payoff_function: Function that takes (my_action, opponent_action) -> payoff
        personality_bias: Bias toward certain action levels (-1 to 1, where 1 = warm bias)
        bias_strength: Strength of personality bias

    Returns:
        Selected continuous action with personality influence
    """

    # Get base action using Thompson sampling
    base_action = select_action_thompson_continuous(agent, payoff_function)

    # Apply personality bias
    bias_target = 0.5 + 0.5 * personality_bias  # Map [-1,1] to [0,1]
    biased_action = (1 - bias_strength) * base_action + bias_strength * bias_target

    # Clip to valid range
    return np.clip(biased_action, 0.0, 1.0)


# Utility functions for action selection analysis
def evaluate_action_quality(
    agent: ContinuousBayesianAgent,
    payoff_function: Callable[[float, float], float],
    selected_action: float,
    true_opponent_action: float = None,
) -> dict:
    """
    Evaluate the quality of a selected action.

    Args:
        agent: ContinuousBayesianAgent object
        payoff_function: Payoff function used
        selected_action: Action that was selected
        true_opponent_action: If known, the opponent's actual action

    Returns:
        Dictionary with action quality metrics
    """

    belief_stats = agent.get_belief_stats()
    expected_opponent = belief_stats["expected_opponent_mean"][0]

    # Expected payoff given belief
    expected_payoff = payoff_function(selected_action, expected_opponent)

    # Actual payoff if opponent action is known
    actual_payoff = None
    if true_opponent_action is not None:
        actual_payoff = payoff_function(selected_action, true_opponent_action)

    # Optimal action given true opponent (if known)
    optimal_action = None
    if true_opponent_action is not None:

        def neg_payoff(my_action):
            return -payoff_function(my_action, true_opponent_action)

        result = minimize_scalar(neg_payoff, bounds=(0, 1), method="bounded")
        optimal_action = result.x

    return {
        "selected_action": selected_action,
        "expected_payoff": expected_payoff,
        "actual_payoff": actual_payoff,
        "optimal_action": optimal_action,
        "regret": (
            (payoff_function(optimal_action, true_opponent_action) - actual_payoff)
            if actual_payoff is not None
            else None
        ),
        "belief_accuracy": (
            abs(expected_opponent - true_opponent_action)
            if true_opponent_action is not None
            else None
        ),
    }
