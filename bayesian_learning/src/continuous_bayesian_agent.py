# src/continuous_bayesian_agent.py
"""
Parametric continuous Bayesian agent for interpersonal dynamics
Uses Normal-Inverse-Wishart conjugate prior for efficient learning
"""

import numpy as np
from scipy.stats import multivariate_normal, invwishart
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class ContinuousBayesianAgent:
    """
    Parametric Bayesian agent for continuous interpersonal actions.

    Assumes opponent actions follow multivariate normal distribution N(μ, Σ).
    Uses Normal-Inverse-Wishart conjugate prior for efficient belief updates.

    Action space: [warmth_level] ∈ [0,1] (can extend to [warmth, dominance])
    """

    agent_id: str = "continuous_agent"
    action_dim: int = 1  # Start with 1D warmth, can extend to 2D [warmth, dominance]

    # Prior beliefs about opponent's action distribution N(μ, Σ)
    # Normal-Inverse-Wishart parameters
    belief_mu_0: np.ndarray = field(
        default_factory=lambda: np.array([0.5])
    )  # Prior mean
    belief_kappa_0: float = 1.0  # Prior precision on mean (lower = more uncertain)
    belief_nu_0: float = 2.0  # Prior degrees of freedom (must be > action_dim)
    belief_psi_0: np.ndarray = field(
        default_factory=lambda: np.array([[0.1]])
    )  # Prior scale matrix

    # Current posterior parameters (updated during interaction)
    belief_mu_n: np.ndarray = field(default_factory=lambda: np.array([0.5]))
    belief_kappa_n: float = 1.0
    belief_nu_n: float = 2.0
    belief_psi_n: np.ndarray = field(default_factory=lambda: np.array([[0.1]]))

    # Psychological parameters
    lambda_loss: float = 1.0
    temperature: float = 1.0

    # History tracking
    action_history: List[np.ndarray] = field(default_factory=list)
    opponent_history: List[np.ndarray] = field(default_factory=list)
    payoff_history: List[float] = field(default_factory=list)

    def __post_init__(self):
        """Initialize posterior parameters with priors"""
        self.belief_mu_n = self.belief_mu_0.copy()
        self.belief_kappa_n = self.belief_kappa_0
        self.belief_nu_n = self.belief_nu_0
        self.belief_psi_n = self.belief_psi_0.copy()

    def update_beliefs(self, opponent_action: np.ndarray) -> None:
        """
        Update beliefs using Normal-Inverse-Wishart conjugate update

        Args:
            opponent_action: Opponent's continuous action(s)
        """

        # Ensure opponent_action is numpy array with correct shape
        opponent_action = np.atleast_1d(opponent_action)
        if len(opponent_action) != self.action_dim:
            raise ValueError(f"Action must have {self.action_dim} dimensions")

        # Store observation
        self.opponent_history.append(opponent_action.copy())
        n = len(self.opponent_history)

        if n == 1:
            # First observation: simple update
            x = opponent_action.reshape(-1, 1)

            # Update parameters
            self.belief_kappa_n = self.belief_kappa_0 + 1
            self.belief_nu_n = self.belief_nu_0 + 1

            self.belief_mu_n = (
                self.belief_kappa_0 * self.belief_mu_0 + opponent_action
            ) / self.belief_kappa_n

            diff = opponent_action - self.belief_mu_0
            self.belief_psi_n = self.belief_psi_0 + (
                self.belief_kappa_0 / self.belief_kappa_n
            ) * np.outer(diff, diff)
        else:
            # Multiple observations: full conjugate update
            X = np.array(self.opponent_history)  # n x d matrix
            x_bar = np.mean(X, axis=0)

            # Updated parameters
            self.belief_kappa_n = self.belief_kappa_0 + n
            self.belief_nu_n = self.belief_nu_0 + n

            self.belief_mu_n = (
                self.belief_kappa_0 * self.belief_mu_0 + n * x_bar
            ) / self.belief_kappa_n

            # Sample covariance matrix
            if n > 1:
                S = np.cov(X, rowvar=False, bias=False) * (
                    n - 1
                )  # Sum of squares matrix
                if self.action_dim == 1:
                    S = S.reshape(1, 1)
            else:
                S = np.zeros((self.action_dim, self.action_dim))

            # Update Psi
            diff = x_bar - self.belief_mu_0
            self.belief_psi_n = (
                self.belief_psi_0
                + S
                + (self.belief_kappa_0 * n / self.belief_kappa_n) * np.outer(diff, diff)
            )

    def sample_opponent_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample opponent's mean and covariance from posterior

        Returns:
            Tuple of (sampled_mean, sampled_covariance)
        """

        # Sample covariance matrix from Inverse-Wishart
        if self.belief_nu_n > self.action_dim + 1:
            try:
                sampled_sigma = invwishart.rvs(
                    df=self.belief_nu_n, scale=self.belief_psi_n
                )
                if self.action_dim == 1:
                    sampled_sigma = sampled_sigma.reshape(1, 1)
            except:
                # Fallback if numerical issues
                sampled_sigma = self.belief_psi_n / (
                    self.belief_nu_n - self.action_dim - 1
                )
        else:
            sampled_sigma = self.belief_psi_n.copy()

        # Sample mean from Normal given sampled covariance
        mean_cov = sampled_sigma / self.belief_kappa_n
        try:
            sampled_mu = multivariate_normal.rvs(mean=self.belief_mu_n, cov=mean_cov)
            sampled_mu = np.atleast_1d(sampled_mu)
        except:
            # Fallback for 1D case
            sampled_mu = np.random.normal(self.belief_mu_n[0], np.sqrt(mean_cov[0, 0]))
            sampled_mu = np.array([sampled_mu])

        return sampled_mu, sampled_sigma

    def predict_opponent_action(self, n_samples: int = 1) -> np.ndarray:
        """
        Predict opponent's next action using Thompson sampling

        Args:
            n_samples: Number of samples to draw

        Returns:
            Sampled opponent actions
        """

        # Sample distribution parameters
        mu_sample, sigma_sample = self.sample_opponent_distribution()

        # Sample actions from the sampled distribution
        if n_samples == 1:
            if self.action_dim == 1:
                action = np.random.normal(mu_sample[0], np.sqrt(sigma_sample[0, 0]))
                return np.clip(np.array([action]), 0, 1)
            else:
                action = multivariate_normal.rvs(mean=mu_sample, cov=sigma_sample)
                return np.clip(action, 0, 1)
        else:
            if self.action_dim == 1:
                actions = np.random.normal(
                    mu_sample[0], np.sqrt(sigma_sample[0, 0]), size=n_samples
                )
                return np.clip(actions.reshape(-1, 1), 0, 1)
            else:
                actions = multivariate_normal.rvs(
                    mean=mu_sample, cov=sigma_sample, size=n_samples
                )
                return np.clip(actions, 0, 1)

    def get_belief_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics of current beliefs about opponent

        Returns:
            Dictionary with belief statistics
        """

        # Posterior mean of opponent's typical behavior
        expected_mu = self.belief_mu_n.copy()

        # Expected covariance matrix
        if self.belief_nu_n > self.action_dim + 1:
            expected_sigma = self.belief_psi_n / (
                self.belief_nu_n - self.action_dim - 1
            )
        else:
            expected_sigma = self.belief_psi_n.copy()

        # Uncertainty in mean estimate
        mean_uncertainty = expected_sigma / self.belief_kappa_n

        stats = {
            "expected_opponent_mean": expected_mu,
            "expected_opponent_cov": expected_sigma,
            "mean_uncertainty": mean_uncertainty,
            "n_observations": len(self.opponent_history),
            "warmth_belief": expected_mu[0],
            "belief_confidence": self.belief_kappa_n
            / self.belief_kappa_0,  # Relative confidence
        }

        if self.action_dim > 1:
            stats["dominance_belief"] = expected_mu[1] if len(expected_mu) > 1 else None

        return stats

    def __str__(self) -> str:
        """String representation of the agent"""
        stats = self.get_belief_stats()
        return (
            f"Continuous Bayesian Agent: {self.agent_id}\n"
            f"Belief about opponent warmth: {stats['warmth_belief']:.3f}\n"
            f"Belief confidence: {stats['belief_confidence']:.2f}x\n"
            f"Actions taken: {len(self.action_history)}"
        )


def create_continuous_bayesian_agent(
    agent_id: str = "agent_1",
    action_dim: int = 1,
    prior_mean: float = 0.5,
    prior_confidence: float = 1.0,
    prior_variance: float = 0.1,
    lambda_loss: float = 1.0,
    temperature: float = 1.0,
) -> ContinuousBayesianAgent:
    """
    Factory function to create continuous Bayesian agent with specified parameters

    Args:
        agent_id: Identifier for the agent
        action_dim: Dimension of action space (1 for warmth only, 2 for warmth+dominance)
        prior_mean: Prior belief about opponent's typical action level
        prior_confidence: How confident we are in prior belief (higher = more confident)
        prior_variance: Expected variance in opponent's actions (lower = more consistent opponent)
        lambda_loss: Loss aversion parameter
        temperature: Temperature for action selection

    Returns:
        Configured ContinuousBayesianAgent
    """

    # Set up prior parameters
    mu_0 = np.full(action_dim, prior_mean)
    psi_0 = np.eye(action_dim) * prior_variance

    return ContinuousBayesianAgent(
        agent_id=agent_id,
        action_dim=action_dim,
        belief_mu_0=mu_0,
        belief_kappa_0=prior_confidence,
        belief_nu_0=action_dim + 1.0,  # Minimum valid value
        belief_psi_0=psi_0,
        lambda_loss=lambda_loss,
        temperature=temperature,
    )
