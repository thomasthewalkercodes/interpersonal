"""
Interaction environment for interpersonal behavior simulation.
Manages the interaction between two agents and their states.
"""

import numpy as np
from typing import Tuple, Optional
from interfaces import InteractionEnvironment, PayoffCalculator, AgentState


class InterpersonalEnvironment(InteractionEnvironment):
    """
    Environment for managing interpersonal interactions between two agents.
    Handles state updates, payoff calculations, and episode management.
    """

    def __init__(
        self,
        payoff_calculator: PayoffCalculator,
        agent1_state: AgentState,
        agent2_state: AgentState,
        agent1_id: str = "agent1",
        agent2_id: str = "agent2",
        max_steps_per_episode: int = 50,
        termination_threshold: float = 0.1,
    ):
        """
        Initialize the interaction environment.

        Args:
            payoff_calculator: Calculator for agent payoffs
            agent1_state: State manager for agent 1
            agent2_state: State manager for agent 2
            agent1_id: Identifier for agent 1
            agent2_id: Identifier for agent 2
            max_steps_per_episode: Maximum steps before episode ends
            termination_threshold: Threshold for early termination (optional)
        """
        self.payoff_calculator = payoff_calculator
        self.agent1_state = agent1_state
        self.agent2_state = agent2_state
        self.agent1_id = agent1_id
        self.agent2_id = agent2_id
        self.max_steps_per_episode = max_steps_per_episode
        self.termination_threshold = termination_threshold

        # Episode tracking
        self.current_step = 0
        self.episode_count = 0

        # Verify state dimensions match
        if agent1_state.get_state_dimension() != agent2_state.get_state_dimension():
            raise ValueError("Agent states must have the same dimension")

        self.state_dim = agent1_state.get_state_dimension()

    def step(
        self, agent1_action: float, agent2_action: float
    ) -> Tuple[np.ndarray, np.ndarray, float, float, bool]:
        """
        Execute one interaction step.

        Args:
            agent1_action: Action taken by agent 1 (warmth level -1 to 1)
            agent2_action: Action taken by agent 2 (warmth level -1 to 1)

        Returns:
            Tuple of (agent1_next_state, agent2_next_state, agent1_reward, agent2_reward, done)
        """
        # Clip actions to valid range
        agent1_action = np.clip(agent1_action, -1.0, 1.0)
        agent2_action = np.clip(agent2_action, -1.0, 1.0)

        # Calculate payoffs
        agent1_payoff, agent2_payoff = self.payoff_calculator.calculate_payoff(
            agent1_action, agent2_action, self.agent1_id, self.agent2_id
        )

        # Update agent states
        self.agent1_state.update_state(agent1_action, agent2_action, agent1_payoff)
        self.agent2_state.update_state(agent2_action, agent1_action, agent2_payoff)

        # Get next states
        next_state1 = self.agent1_state.get_state_vector()
        next_state2 = self.agent2_state.get_state_vector()

        # Update step counter
        self.current_step += 1

        # Check for episode termination
        done = self._check_termination(agent1_action, agent2_action)

        return next_state1, next_state2, agent1_payoff, agent2_payoff, done

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reset environment and return initial states for both agents.

        Returns:
            Tuple of (agent1_initial_state, agent2_initial_state)
        """
        # Reset agent states
        self.agent1_state.reset_state()
        self.agent2_state.reset_state()

        # Reset episode tracking
        self.current_step = 0
        self.episode_count += 1

        # Get initial states
        initial_state1 = self.agent1_state.get_state_vector()
        initial_state2 = self.agent2_state.get_state_vector()

        return initial_state1, initial_state2

    def get_state_dimension(self) -> int:
        """Return the dimension of the state space."""
        return self.state_dim

    def _check_termination(self, agent1_action: float, agent2_action: float) -> bool:
        """
        Check if episode should terminate.

        Args:
            agent1_action: Recent action from agent 1
            agent2_action: Recent action from agent 2

        Returns:
            True if episode should end
        """
        # Episode ends if max steps reached
        if self.current_step >= self.max_steps_per_episode:
            return True

        # Optional: Early termination if both agents reach extreme negative warmth
        # This represents a complete breakdown in the relationship
        if (
            agent1_action < -0.9
            and agent2_action < -0.9
            and hasattr(self.agent1_state, "get_trust_level")
            and hasattr(self.agent2_state, "get_trust_level")
        ):

            if (
                self.agent1_state.get_trust_level() < -0.8
                and self.agent2_state.get_trust_level() < -0.8
            ):
                return True

        return False

    def get_interaction_stats(self) -> dict:
        """Get statistics about the current interaction."""
        stats = {
            "current_step": self.current_step,
            "episode_count": self.episode_count,
            "max_steps": self.max_steps_per_episode,
        }

        # Add agent-specific stats if available
        if hasattr(self.agent1_state, "get_trust_level"):
            stats["agent1_trust"] = self.agent1_state.get_trust_level()
            stats["agent1_satisfaction"] = self.agent1_state.get_satisfaction_level()

        if hasattr(self.agent2_state, "get_trust_level"):
            stats["agent2_trust"] = self.agent2_state.get_trust_level()
            stats["agent2_satisfaction"] = self.agent2_state.get_satisfaction_level()

        return stats


class SimplePayoffCalculator(PayoffCalculator):
    """
    Simple implementation of payoff calculator.
    You can replace this with your own implementation.
    """

    def __init__(
        self,
        cooperation_bonus: float = 5.0,
        betrayal_penalty: float = -3.0,
        neutral_payoff: float = 1.0,
    ):
        """
        Initialize payoff calculator.

        Args:
            cooperation_bonus: Bonus for mutual cooperation (both positive warmth)
            betrayal_penalty: Penalty for betrayal (one negative, one positive)
            neutral_payoff: Baseline payoff for neutral interactions
        """
        self.cooperation_bonus = cooperation_bonus
        self.betrayal_penalty = betrayal_penalty
        self.neutral_payoff = neutral_payoff

    def calculate_payoff(
        self, agent1_action: float, agent2_action: float, agent1_id: str, agent2_id: str
    ) -> Tuple[float, float]:
        """
        Calculate payoffs based on a simple game theory matrix.

        This is a placeholder implementation. Replace with your actual payoff function.
        """
        # Normalize actions to [0, 1] for easier calculation
        norm_action1 = (agent1_action + 1) / 2
        norm_action2 = (agent2_action + 1) / 2

        # Base payoff from own action (showing warmth is slightly costly)
        base_payoff1 = self.neutral_payoff - 0.1 * norm_action1
        base_payoff2 = self.neutral_payoff - 0.1 * norm_action2

        # Interaction effects
        if agent1_action > 0 and agent2_action > 0:
            # Mutual cooperation - both benefit
            cooperation_benefit = self.cooperation_bonus * min(
                norm_action1, norm_action2
            )
            payoff1 = base_payoff1 + cooperation_benefit
            payoff2 = base_payoff2 + cooperation_benefit

        elif agent1_action < 0 and agent2_action < 0:
            # Mutual defection - both suffer
            defection_penalty = self.betrayal_penalty * min(
                abs(agent1_action), abs(agent2_action)
            )
            payoff1 = base_payoff1 + defection_penalty
            payoff2 = base_payoff2 + defection_penalty

        else:
            # Mixed interaction - exploiter benefits, cooperator suffers
            if agent1_action > agent2_action:
                # Agent 1 is warmer, Agent 2 exploits
                payoff1 = base_payoff1 + self.betrayal_penalty * 0.5
                payoff2 = base_payoff2 + self.cooperation_bonus * 0.3
            else:
                # Agent 2 is warmer, Agent 1 exploits
                payoff1 = base_payoff1 + self.cooperation_bonus * 0.3
                payoff2 = base_payoff2 + self.betrayal_penalty * 0.5

        return payoff1, payoff2


class AdvancedPayoffCalculator(PayoffCalculator):
    """
    More sophisticated payoff calculator that considers relationship history.
    """

    def __init__(self, base_matrix: np.ndarray = None):
        """
        Initialize with a payoff matrix.

        Args:
            base_matrix: 2x2 matrix defining payoffs for [cooperate, defect] actions
        """
        if base_matrix is None:
            # Default prisoner's dilemma-like matrix
            # Format: [[CC, CD], [DC, DD]] where first letter is agent1, second is agent2
            self.base_matrix = np.array(
                [
                    [
                        3,
                        -1,
                    ],  # Agent1 cooperates: gets 3 if agent2 cooperates, -1 if defects
                    [5, 0],  # Agent1 defects: gets 5 if agent2 cooperates, 0 if defects
                ]
            )
        else:
            self.base_matrix = base_matrix

        # Track interaction history for relationship effects
        self.interaction_history = {}

    def calculate_payoff(
        self, agent1_action: float, agent2_action: float, agent1_id: str, agent2_id: str
    ) -> Tuple[float, float]:
        """
        Calculate payoffs using continuous actions mapped to cooperation/defection probabilities.
        """
        # Convert continuous actions to cooperation probabilities
        coop_prob1 = (agent1_action + 1) / 2  # Map [-1,1] to [0,1]
        coop_prob2 = (agent2_action + 1) / 2

        # Calculate expected payoffs based on the matrix
        # Agent 1's payoff
        payoff1 = (
            coop_prob1 * coop_prob2 * self.base_matrix[0, 0]
            + coop_prob1 * (1 - coop_prob2) * self.base_matrix[0, 1]
            + (1 - coop_prob1) * coop_prob2 * self.base_matrix[1, 0]
            + (1 - coop_prob1) * (1 - coop_prob2) * self.base_matrix[1, 1]
        )

        # Agent 2's payoff (transpose the matrix)
        payoff2 = (
            coop_prob2 * coop_prob1 * self.base_matrix[0, 0]
            + coop_prob2 * (1 - coop_prob1) * self.base_matrix[1, 0]
            + (1 - coop_prob2) * coop_prob1 * self.base_matrix[0, 1]
            + (1 - coop_prob2) * (1 - coop_prob1) * self.base_matrix[1, 1]
        )

        # Store interaction for future relationship effects
        pair_key = tuple(sorted([agent1_id, agent2_id]))
        if pair_key not in self.interaction_history:
            self.interaction_history[pair_key] = []

        self.interaction_history[pair_key].append((agent1_action, agent2_action))

        # Relationship bonus/penalty based on history
        relationship_modifier = self._calculate_relationship_modifier(pair_key)

        return payoff1 + relationship_modifier, payoff2 + relationship_modifier

    def _calculate_relationship_modifier(self, pair_key: tuple) -> float:
        """Calculate relationship modifier based on interaction history."""
        history = self.interaction_history.get(pair_key, [])

        if len(history) < 2:
            return 0.0

        # Calculate consistency bonus
        recent_actions = history[-5:]  # Look at last 5 interactions

        if len(recent_actions) >= 2:
            # Reward consistent mutual cooperation
            mutual_coop_count = sum(1 for a1, a2 in recent_actions if a1 > 0 and a2 > 0)
            consistency_bonus = 0.5 * (mutual_coop_count / len(recent_actions))

            # Penalize consistent mutual defection
            mutual_defect_count = sum(
                1 for a1, a2 in recent_actions if a1 < -0.5 and a2 < -0.5
            )
            consistency_penalty = -0.3 * (mutual_defect_count / len(recent_actions))

            return consistency_bonus + consistency_penalty

        return 0.0
