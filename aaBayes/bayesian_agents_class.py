"""
Agent class for the interpersonal circumplex model with Bayesian reasoning.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from circumplex_model_core_functions import (
    CircumplexSpace,
    BayesianBelief,
    PayoffMatrix,
)


class Agent:
    """
    An agent that interacts using the interpersonal circumplex model.
    Features:
    - Bayesian beliefs about other agents
    - Perception bias
    - Adaptive behavior based on feedback
    - Mood tracking
    """

    def __init__(self, agent_id: str, personality: Dict[str, Any]):
        """
        Initialize an agent with personality parameters.

        Args:
            agent_id: Unique identifier
            personality: Dictionary containing:
                - base_behavior: (angle, radius) default behavior
                - warmth_bias: Perception bias for warmth (-1 to 1)
                - dominance_bias: Perception bias for dominance (-1 to 1)
                - learning_rate: How quickly beliefs update (0-1)
                - adaptation_rate: How quickly behavior changes (0-1)
                - mood_sensitivity: How much mood affects behavior (0-1)
                - risk_aversion: Preference for safe vs risky behaviors (0-1)
                - initial_mood: Starting mood level (-1 to 1)
        """
        self.id = agent_id
        self.personality = personality

        # Current behavior (angle, radius)
        self.current_behavior = personality["base_behavior"]
        self.base_behavior = personality["base_behavior"]

        # Mood tracking
        self.mood = personality.get("initial_mood", 0.0)
        self.mood_history = [self.mood]

        # Perception biases
        self.warmth_bias = personality.get("warmth_bias", 0.0)
        self.dominance_bias = personality.get("dominance_bias", 0.0)

        # Learning parameters
        self.learning_rate = personality.get("learning_rate", 0.1)
        self.adaptation_rate = personality.get("adaptation_rate", 0.1)
        self.mood_sensitivity = personality.get("mood_sensitivity", 0.5)
        self.risk_aversion = personality.get("risk_aversion", 0.5)

        # Beliefs about other agents
        self.beliefs = {}

        # Interaction history
        self.interaction_history = []
        self.payoff_history = []

    def initialize_belief(
        self, other_agent_id: str, initial_belief: Optional[Dict] = None
    ):
        """Initialize Bayesian belief about another agent."""
        if initial_belief:
            mean = initial_belief.get("mean", np.array([0.0, 0.0]))
            cov = initial_belief.get("cov", np.array([[0.5, 0.0], [0.0, 0.5]]))
        else:
            # Default: neutral with high uncertainty
            mean = np.array([0.0, 0.0])
            cov = np.array([[0.5, 0.0], [0.0, 0.5]])

        self.beliefs[other_agent_id] = BayesianBelief(mean, cov)

    def perceive_behavior(
        self, actual_behavior: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Apply perception bias to observed behavior.

        Args:
            actual_behavior: (angle, radius) of actual behavior

        Returns:
            Perceived (angle, radius) after applying bias
        """
        angle, radius = actual_behavior

        # Convert to Cartesian for bias application
        warmth, dominance = CircumplexSpace.circumplex_to_cartesian(angle, radius)

        # Apply perception biases
        # Linear shift with sigmoid bounding to keep values reasonable
        perceived_warmth = warmth + self.warmth_bias * (1 - abs(warmth))
        perceived_dominance = dominance + self.dominance_bias * (1 - abs(dominance))

        # Ensure values stay within reasonable bounds
        perceived_warmth = np.clip(perceived_warmth, -1, 1)
        perceived_dominance = np.clip(perceived_dominance, -1, 1)

        # Convert back to circumplex
        return CircumplexSpace.cartesian_to_circumplex(
            perceived_warmth, perceived_dominance
        )

    def choose_behavior(
        self, other_agent_id: str, n_samples: int = 100
    ) -> Tuple[float, float]:
        """
        Choose behavior based on Bayesian reasoning about the other agent.

        Uses expected utility maximization considering:
        - Beliefs about other agent's likely behaviors
        - Risk aversion
        - Current mood
        """
        if other_agent_id not in self.beliefs:
            # If no belief exists, use base behavior
            return self.current_behavior

        belief = self.beliefs[other_agent_id]

        # Sample possible behaviors of the other agent
        other_samples = belief.sample_beliefs(n_samples)

        # Generate candidate behaviors (variations around current behavior)
        candidate_behaviors = self._generate_candidate_behaviors()

        # Calculate expected utility for each candidate
        payoff_calculator = PayoffMatrix()
        best_behavior = self.current_behavior
        best_expected_utility = -float("inf")

        for candidate in candidate_behaviors:
            expected_utility = 0
            utility_variance = 0

            for other_behavior in other_samples:
                payoff = payoff_calculator.calculate_payoff(
                    candidate, (other_behavior[0], other_behavior[1])
                )
                expected_utility += payoff / n_samples
                utility_variance += (payoff**2) / n_samples

            utility_variance -= expected_utility**2

            # Risk-adjusted utility (higher risk_aversion penalizes variance)
            risk_adjusted_utility = expected_utility - self.risk_aversion * np.sqrt(
                utility_variance
            )

            # Mood adjustment (bad mood makes agent less cooperative)
            mood_adjusted_utility = risk_adjusted_utility * (1 + 0.2 * self.mood)

            if mood_adjusted_utility > best_expected_utility:
                best_expected_utility = mood_adjusted_utility
                best_behavior = candidate

        return best_behavior

    def _generate_candidate_behaviors(self, n_candidates: int = 20) -> list:
        """Generate candidate behaviors to consider."""
        candidates = []

        current_angle, current_radius = self.current_behavior

        # Include current behavior
        candidates.append(self.current_behavior)

        # Generate variations
        for _ in range(n_candidates - 1):
            # Vary angle (with larger variations if mood is bad)
            angle_std = 30 * (1 + abs(self.mood))
            new_angle = (current_angle + np.random.normal(0, angle_std)) % 360

            # Vary radius
            radius_std = 0.1 * (1 + abs(self.mood) * 0.5)
            new_radius = np.clip(
                current_radius + np.random.normal(0, radius_std), 0.1, 1.0
            )

            candidates.append((new_angle, new_radius))

        return candidates

    def update_after_interaction(
        self, other_agent_id: str, observed_behavior: Tuple[float, float], payoff: float
    ):
        """
        Update beliefs, mood, and behavior after an interaction.

        Args:
            other_agent_id: ID of the other agent
            observed_behavior: Actual behavior shown by other agent
            payoff: Payoff received from the interaction
        """
        # Perceive the behavior (with bias)
        perceived_behavior = self.perceive_behavior(observed_behavior)

        # Update beliefs about the other agent
        if other_agent_id in self.beliefs:
            self.beliefs[other_agent_id].update(perceived_behavior, self.learning_rate)

        # Update mood based on payoff
        max_payoff = 10.0  # Should match PayoffMatrix max
        normalized_payoff = (payoff / max_payoff) - 0.5  # Center around 0
        mood_delta = normalized_payoff * self.mood_sensitivity
        self.mood = np.clip(self.mood + mood_delta, -1.0, 1.0)
        self.mood_history.append(self.mood)

        # Update behavior based on feedback
        if payoff > 5.0:  # Threshold for "good" interaction
            # Reinforce current behavior (move slightly towards it)
            self._reinforce_behavior()
        else:
            # Adapt behavior (move away from current)
            self._adapt_behavior(perceived_behavior)

        # Store history
        self.interaction_history.append(
            {
                "other_agent": other_agent_id,
                "own_behavior": self.current_behavior,
                "perceived_other": perceived_behavior,
                "actual_other": observed_behavior,
                "payoff": payoff,
                "mood": self.mood,
            }
        )
        self.payoff_history.append(payoff)

    def _reinforce_behavior(self):
        """Reinforce current behavior by reducing exploration."""
        # Move current behavior slightly towards base behavior
        # This creates stability when things are going well
        angle, radius = self.current_behavior
        base_angle, base_radius = self.base_behavior

        # Weighted average with base behavior
        reinforcement_weight = 0.1 * self.adaptation_rate
        new_angle = (
            angle * (1 - reinforcement_weight) + base_angle * reinforcement_weight
        )
        new_radius = (
            radius * (1 - reinforcement_weight) + base_radius * reinforcement_weight
        )

        self.current_behavior = (new_angle % 360, np.clip(new_radius, 0.1, 1.0))

    def _adapt_behavior(self, other_behavior: Tuple[float, float]):
        """Adapt behavior when current strategy isn't working."""
        current_angle, current_radius = self.current_behavior
        other_angle, other_radius = other_behavior

        # Calculate what would have been optimal
        optimal_angle = CircumplexSpace.compute_optimal_match(other_angle)

        # Move towards optimal with some noise for exploration
        adaptation_strength = self.adaptation_rate * (1 + abs(self.mood))

        # Angular adaptation
        angle_diff = CircumplexSpace.angular_distance(current_angle, optimal_angle)
        if angle_diff > 0:
            # Move towards optimal
            if (optimal_angle - current_angle + 360) % 360 < 180:
                new_angle = current_angle + angle_diff * adaptation_strength
            else:
                new_angle = current_angle - angle_diff * adaptation_strength
        else:
            new_angle = current_angle

        # Add exploration noise
        new_angle += np.random.normal(0, 10 * abs(self.mood))
        new_angle = new_angle % 360

        # Radius adaptation (move towards other's radius)
        radius_diff = other_radius - current_radius
        new_radius = current_radius + radius_diff * adaptation_strength * 0.5
        new_radius = np.clip(new_radius, 0.1, 1.0)

        self.current_behavior = (new_angle, new_radius)

    def reset(self):
        """Reset agent to initial state."""
        self.current_behavior = self.base_behavior
        self.mood = self.personality.get("initial_mood", 0.0)
        self.mood_history = [self.mood]
        self.beliefs = {}
        self.interaction_history = []
        self.payoff_history = []
