from dataclasses import dataclass


@dataclass
class agent_generator:
    """Defines a specific type of opponent behavior"""

    name: str
    description: str
    learning_rate: float
    prior_expectation: float
    prior_strength: float
    risk_sensitivity: float
    exploration_rate: float


# Standard opponent type definitions
agent_personalities = {
    "COLD_RIGID": agent_personalities(
        name="Cold Rigid",
        description="Consistently cold with strong priors",
        learning_rate=0.1,
        prior_expectation=0.2,
        prior_strength=40.0,
        risk_sensitivity=0.7,
        exploration_rate=0.1,
    ),
    "WARM_FLEXIBLE": agent_personalities(
        name="Warm Flexible",
        description="Initially warm but adapts quickly",
        learning_rate=0.3,
        prior_expectation=0.8,
        prior_strength=10.0,
        risk_sensitivity=0.3,
        exploration_rate=0.2,
    ),
}
