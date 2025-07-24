"""
Advanced agent personality types for interpersonal behavior research.
These extend your base system with psychologically-inspired agent types.
"""

from agent_configuration import BaseAgentConfig
import numpy as np


class NarcissisticAgentConfig(BaseAgentConfig):
    """Narcissistic agents: High initial self-regard, exploitative, poor empathy."""

    def __init__(self, **kwargs):
        defaults = {
            "lr_actor": 8e-4,  # Learn quickly to exploit
            "lr_critic": 8e-4,
            "gamma": 0.99,  # Very forward-looking (self-interested planning)
            "alpha": 0.4,  # High exploration to find exploitative strategies
            "initial_trust": -0.1,  # Slightly distrustful initially
            "initial_satisfaction": 0.7,  # High baseline self-satisfaction
            "memory_length": 20,  # Short memory - don't hold grudges, just exploit
            "noise_scale": 0.2,  # High variability in behavior
            "tau": 0.01,  # Fast adaptation to new exploitation opportunities
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


class BorderlineAgentConfig(BaseAgentConfig):
    """Borderline agents: Intense relationships, fear of abandonment, emotional volatility."""

    def __init__(self, **kwargs):
        defaults = {
            "lr_actor": 1e-3,  # Very fast learning (emotional reactivity)
            "lr_critic": 1e-3,
            "gamma": 0.85,  # Less forward-looking (impulsive)
            "alpha": 0.6,  # Very high exploration (emotional volatility)
            "initial_trust": 0.8,  # Start with high trust (idealization)
            "initial_satisfaction": -0.2,  # Low baseline satisfaction
            "memory_length": 150,  # Very long memory (remember every slight)
            "noise_scale": 0.3,  # High behavioral volatility
            "tau": 0.02,  # Fast emotional updates
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


class AvoidantAgentConfig(BaseAgentConfig):
    """Avoidant agents: Uncomfortable with closeness, self-reliant, emotionally distant."""

    def __init__(self, **kwargs):
        defaults = {
            "lr_actor": 5e-5,  # Very slow learning (resistance to change)
            "lr_critic": 5e-5,
            "gamma": 0.99,  # Long-term thinking but emotionally distant
            "alpha": 0.02,  # Very low exploration (predictable, distant)
            "initial_trust": -0.5,  # Low initial trust
            "initial_satisfaction": 0.2,  # Moderate self-satisfaction from independence
            "memory_length": 80,  # Moderate memory
            "noise_scale": 0.01,  # Very low noise (consistent emotional distance)
            "tau": 0.0005,  # Extremely slow updates (emotional rigidity)
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


class EmpathicAgentConfig(BaseAgentConfig):
    """Highly empathic agents: Mirror partner's emotions, prosocial, sometimes self-sacrificing."""

    def __init__(self, **kwargs):
        defaults = {
            "lr_actor": 6e-4,  # Moderate-fast learning (responsive to others)
            "lr_critic": 6e-4,
            "gamma": 0.95,  # Slightly less forward-looking (focus on immediate social harmony)
            "alpha": 0.15,  # Lower exploration (prefer known prosocial strategies)
            "initial_trust": 0.6,  # High initial trust
            "initial_satisfaction": 0.1,  # Lower self-focus
            "memory_length": 60,  # Good memory for partner's needs
            "noise_scale": 0.08,  # Low noise (consistent empathic responses)
            "tau": 0.008,  # Fast adaptation to partner's emotional state
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


class MachinavellianeAgentConfig(BaseAgentConfig):
    """Machiavellian agents: Strategic manipulation, long-term planning, charm when useful."""

    def __init__(self, **kwargs):
        defaults = {
            "lr_actor": 2e-4,  # Slower learning (calculated, not reactive)
            "lr_critic": 2e-4,
            "gamma": 0.995,  # Very long-term planning
            "alpha": 0.12,  # Moderate exploration (calculated risks)
            "initial_trust": 0.0,  # Neutral - assess first
            "initial_satisfaction": 0.4,  # Moderate self-satisfaction
            "memory_length": 200,  # Very long memory for strategic planning
            "noise_scale": 0.05,  # Low noise (controlled behavior)
            "tau": 0.003,  # Slow, deliberate updates
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


class SecureAgentConfig(BaseAgentConfig):
    """Securely attached agents: Balanced, resilient, good emotional regulation."""

    def __init__(self, **kwargs):
        defaults = {
            "lr_actor": 3e-4,  # Balanced learning rate
            "lr_critic": 3e-4,
            "gamma": 0.97,  # Balanced temporal perspective
            "alpha": 0.2,  # Moderate exploration
            "initial_trust": 0.3,  # Moderate initial trust (realistic optimism)
            "initial_satisfaction": 0.4,  # Healthy self-regard
            "memory_length": 50,  # Balanced memory
            "noise_scale": 0.1,  # Moderate variability
            "tau": 0.005,  # Balanced adaptation rate
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


class ReactiveAgentConfig(BaseAgentConfig):
    """Reactive agents: Tit-for-tat style, quick to respond to partner's behavior."""

    def __init__(self, **kwargs):
        defaults = {
            "lr_actor": 1e-3,  # Fast learning (quick reactions)
            "lr_critic": 1e-3,
            "gamma": 0.9,  # Less forward-looking (reactive to immediate)
            "alpha": 0.25,  # Moderate-high exploration
            "initial_trust": 0.2,  # Slightly optimistic but cautious
            "initial_satisfaction": 0.3,
            "memory_length": 30,  # Shorter memory (focus on recent interactions)
            "noise_scale": 0.15,  # Higher noise (reactive volatility)
            "tau": 0.015,  # Fast updates to partner changes
        }
        defaults.update(kwargs)
        super().__init__(**defaults)
