import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
from Reinforcement_learning import (
    LearningConfig,
    WarmthActionSpace,
    WarmthLearningAgent,
)
from newmodelplot import plot_simulation_results


def calculate_warmth_payoff(
    w1: float, w2: float, alpha: float = 4, beta: float = 20
) -> float:
    """Calculate payoff for warmth interaction between two agents."""
    # Ensure inputs are in valid range
    w1 = np.clip(w1, 0, 1)
    w2 = np.clip(w2, 0, 1)

    # Calculate payoff components
    mismatch = (w1 - w2) ** 2
    risk_factor = w1
    penalty = risk_factor * mismatch * beta
    base_payoff = np.exp(-alpha * mismatch)

    return base_payoff - penalty


@dataclass
class SimulationConfig:
    """Configuration for simulation parameters"""

    n_rounds: int = 1000
    n_bins: int = 20
    save_history: bool = True
    plot_results: bool = True


def create_agents(
    sim_config: SimulationConfig,
) -> Tuple[WarmthLearningAgent, WarmthLearningAgent]:
    """Create and configure agents with different prior expectations"""
    # Create learning configs
    config1 = LearningConfig(
        alpha=3,  # Learning rate
        epsilon=0.5,  # Exploration rate
        risk_sensitivity=0.1,  # Moderate risk sensitivity
        prior_expectation=0.5,  # Expects cold behavior
        prior_strength=0.8,  # Strong prior beliefs - slower to adapt
    )
    config2 = LearningConfig(
        alpha=3,  # Faster learning rate
        epsilon=0.5,  # Same exploration
        risk_sensitivity=0.1,  # Less risk-sensitive
        prior_expectation=0.8,  # Expects warm behavior
        prior_strength=0.5,  # Weaker prior beliefs - faster to adapt
    )

    # Create action space
    action_space = WarmthActionSpace(n_bins=sim_config.n_bins)

    # Create agents
    agent1 = WarmthLearningAgent(config1, action_space, "Agent1")
    agent2 = WarmthLearningAgent(config2, action_space, "Agent2")

    return agent1, agent2


def run_simulation(
    agent1: WarmthLearningAgent,
    agent2: WarmthLearningAgent,
    sim_config: SimulationConfig,
) -> Tuple[List[float], List[float]]:
    """Run simulation for specified number of rounds"""
    payoffs1 = []
    payoffs2 = []
    actions1 = []
    actions2 = []

    for round_num in range(sim_config.n_rounds):
        # Get actions
        w1 = agent1.choose_action()
        w2 = agent2.choose_action()

        # Store actions
        actions1.append(w1)
        actions2.append(w2)

        # Calculate payoffs
        payoff1 = calculate_warmth_payoff(w1, w2)
        payoff2 = calculate_warmth_payoff(w2, w1)

        # Update agents
        state1 = agent1.action_space.discretize(w1)
        state2 = agent2.action_space.discretize(w2)

        next_state1 = agent1.action_space.discretize(w2)
        next_state2 = agent2.action_space.discretize(w1)

        agent1.update(state1, w1, payoff1, next_state1)
        agent2.update(state2, w2, payoff2, next_state2)

        payoffs1.append(payoff1)
        payoffs2.append(payoff2)

    # Store action histories in agents
    agent1.action_history = actions1
    agent2.action_history = actions2

    return payoffs1, payoffs2


def main():
    """Main function to run simulation"""
    # Create simulation configuration
    sim_config = SimulationConfig()

    # Create agents
    agent1, agent2 = create_agents(sim_config)

    # Run simulation
    payoffs1, payoffs2 = run_simulation(agent1, agent2, sim_config)

    # Use new plotting function if plot_results is True
    if sim_config.plot_results:
        fig = plot_simulation_results(payoffs1, payoffs2, agent1, agent2)
        plt.show()


if __name__ == "__main__":
    # Only show plot when running this file directly
    main()
