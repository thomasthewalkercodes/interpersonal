# src/bayesian_agent.py
"""
Core Bayesian agent implementation for interpersonal dynamics
"""

import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class BayesianAgent:
    """
    Bayesian learning agent that maintains beliefs about opponent's
    action probabilities using Beta distributions

    Attributes:
        agent_id: String identifying the agent
        belief_alpha: Prior count for warm actions (default: 1.0)
        belief_beta: Prior count for cold actions (default: 1.0)
        lambda_loss: Loss aversion parameter (default: 1.0)
        temperature: Softmax temperature for action selection (default: 1.0)
        action_history: List of agent's own actions
        opponent_history: List of observed opponent actions
        payoff_history: List of received payoffs
        opponent_warm_prob: Current estimate of opponent's warmth probability
    """

    agent_id: str = "agent_1"
    belief_alpha: float = 1.0
    belief_beta: float = 1.0
    lambda_loss: float = 1.0
    temperature: float = 1.0

    # History tracking (initialized as empty lists)
    action_history: List[int] = field(default_factory=list)
    opponent_history: List[int] = field(default_factory=list)
    payoff_history: List[float] = field(default_factory=list)

    # Action space (binary: 0=Cold, 1=Warm)
    action_space: List[int] = field(default_factory=lambda: [0, 1])

    def __post_init__(self):
        """Calculate initial opponent warm probability after initialization"""
        self.opponent_warm_prob = self.belief_alpha / (
            self.belief_alpha + self.belief_beta
        )

    def __str__(self) -> str:
        """String representation of the agent"""
        return (
            f"Bayesian Agent: {self.agent_id}\n"
            f"Belief about opponent warmth: Beta({self.belief_alpha:.2f}, {self.belief_beta:.2f})\n"
            f"Estimated opponent warm probability: {self.opponent_warm_prob:.3f}\n"
            f"Actions taken: {len(self.action_history)}"
        )

    def get_belief_stats(self) -> Dict[str, Any]:
        """
        Returns comprehensive statistics about agent's current beliefs

        Returns:
            Dictionary containing belief statistics including mean, variance,
            credible intervals, and observation counts
        """
        alpha = self.belief_alpha
        beta = self.belief_beta
        total = alpha + beta

        # Beta distribution statistics
        mean_belief = alpha / total
        variance = (alpha * beta) / (total**2 * (total + 1))

        # 95% credible interval using beta quantiles
        from scipy.stats import beta as beta_dist

        lower_ci = beta_dist.ppf(0.025, alpha, beta)
        upper_ci = beta_dist.ppf(0.975, alpha, beta)

        return {
            "mean": mean_belief,
            "variance": variance,
            "std_dev": np.sqrt(variance),
            "credible_interval_95": [lower_ci, upper_ci],
            "alpha": alpha,
            "beta": beta,
            "total_observations": len(self.opponent_history),
        }


def create_bayesian_agent(
    agent_id: str = "agent_1",
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
    lambda_loss: float = 1.0,
    temperature: float = 1.0,
) -> BayesianAgent:
    """
    Factory function to create a Bayesian learning agent

    Args:
        agent_id: Character string identifying the agent
        alpha_prior: Prior count for warm actions (default: 1)
        beta_prior: Prior count for cold actions (default: 1)
        lambda_loss: Loss aversion parameter (default: 1.0)
        temperature: Softmax temperature for action selection (default: 1.0)

    Returns:
        BayesianAgent object with specified parameters
    """

    return BayesianAgent(
        agent_id=agent_id,
        belief_alpha=alpha_prior,
        belief_beta=beta_prior,
        lambda_loss=lambda_loss,
        temperature=temperature,
    )
