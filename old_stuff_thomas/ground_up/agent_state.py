"""
Fixed version of your agent_state.py with all syntax errors corrected.
Save this as agent_state.py in your directory.
"""

import numpy as np
from collections import deque
from interfaces import AgentState


class InterpersonalAgentState(AgentState):
    def __init__(
        self,
        memory_length: int = 50,
        initial_trust: float = 0.0,
        initial_satisfaction: float = 0.0,
    ):
        self.memory_length = memory_length
        self.initial_trust = initial_trust
        self.initial_satisfaction = initial_satisfaction

        # Internal states
        self.trust_level = initial_trust
        self.satisfaction_level = initial_satisfaction
        self.cumulative_payoff = 0.0
        self.interaction_count = 0

        # Action history
        self.self_action_history = deque(maxlen=memory_length)
        self.other_action_history = deque(maxlen=memory_length)

        # Initialize with neutral actions if memory is empty
        self._initialize_histories()

    def _initialize_histories(self):
        for _ in range(self.memory_length):
            self.self_action_history.append(0.0)
            self.other_action_history.append(0.0)

    def get_state_vector(self) -> np.ndarray:
        """Returning current state as a vector
        State includes:
        - Trust level
        - Satisfaction level
        - average cumulative payoff
        - own action history (last few memory_length actions)
        - Partners action history (last few memory_length actions)
        - recent trend indicators"""

        internal_states = [
            self.trust_level,
            self.satisfaction_level,
            self.cumulative_payoff
            / max(1, self.interaction_count),  # Avoid division by zero
            float(self.interaction_count) / 1000.0,
        ]

        # action histories
        self_history = list(self.self_action_history)
        other_history = list(self.other_action_history)

        # Recent trends (last 10 actions if available)
        recent_window = min(10, len(other_history))
        if recent_window > 0:
            recent_other = other_history[-recent_window:]
            recent_self = self_history[-recent_window:]

            other_trend = np.mean(recent_other) if recent_other else 0.0
            self_trend = np.mean(recent_self) if recent_self else 0.0
            other_volatility = np.std(recent_other) if len(recent_other) > 1 else 0.0
        else:
            other_trend = 0.0
            self_trend = 0.0
            other_volatility = 0.0

        trend_features = [other_trend, self_trend, other_volatility]

        # Combine all features
        state_vector = np.array(
            internal_states + self_history + other_history + trend_features,
            dtype=np.float32,
        )

        return state_vector

    def update_state(self, my_action: float, other_action: float, payoff: float):
        """Update the agent's state based on the action taken and the received payoff
        Args:
        - my_action: own action
        - other_action: partner action
        - payoff: received payoff"""

        # Update action histories
        self.self_action_history.append(my_action)
        self.other_action_history.append(other_action)

        # Update payoff tracking
        self.cumulative_payoff += payoff
        self.interaction_count += 1

        # Update trust based on other's action and consistency
        self._update_trust(other_action)

        # Update satisfaction based on payoff and expectations
        self._update_satisfaction(payoff, my_action, other_action)

    def _update_trust(self, other_action: float):
        """Update trust level based on partner's action"""
        # Trust increases with positive other actions and consistency
        other_positivity = (other_action + 1) / 2  # normalizing

        # Calculate consistency (lower variance = higher consistency)
        if len(self.other_action_history) > 5:
            recent_actions = list(self.other_action_history)[-5:]
            consistency = 1.0 - min(1.0, np.std(recent_actions))
        else:
            consistency = 0.5

        # Trust update with learning rate
        trust_update = 0.1 * (other_positivity * consistency - 0.5)
        self.trust_level = np.clip(self.trust_level + trust_update, -1.0, 1.0)

    def _update_satisfaction(
        self, payoff: float, my_action: float, other_action: float
    ):
        """Update satisfaction based on received payoff and actions"""
        # satisfaction increases with positive payoff and alignment
        # also influenced by mutual cooperation
        cooperation_bonus = 0.0
        if my_action > 0 and other_action > 0:
            cooperation_bonus = 0.2 * min(my_action, other_action)

        # Normalize payoff (adjust as needed)
        normalized_payoff = np.clip(payoff / 10.0, -1.0, 1.0)

        satisfaction_update = 0.15 * (normalized_payoff + cooperation_bonus)
        self.satisfaction_level = np.clip(
            self.satisfaction_level + satisfaction_update, -1.0, 1.0
        )

    def reset_state(self):
        """Reset the agent's state to initial values"""
        self.trust_level = self.initial_trust
        self.satisfaction_level = self.initial_satisfaction
        self.cumulative_payoff = 0.0
        self.interaction_count = 0

        # Reset histories
        self.self_action_history.clear()
        self.other_action_history.clear()
        self._initialize_histories()

    def get_state_dimension(self) -> int:
        """Return the dimension of the state vector"""
        # state vector includes:
        # - Trust level
        # - Satisfaction level
        # - Average cumulative payoff
        # - Interaction count (normalized)
        # - Own action history
        # - Partner's action history
        # - Recent trends (partner trend, self trend, partner volatility)
        return (
            4 + (2 * self.memory_length) + 3
        )  # 4 internal states + 2 * memory length + 3 trend features

    def get_trust_level(self) -> float:
        """Get the current trust level"""
        return self.trust_level

    def get_satisfaction_level(self) -> float:
        """Get the current satisfaction level"""
        return self.satisfaction_level

    def get_average_payoff(self) -> float:
        """Get the cumulative payoff"""
        return self.cumulative_payoff / max(
            1, self.interaction_count
        )  # Avoid division by zero
