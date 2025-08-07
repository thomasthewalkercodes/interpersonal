# src/continuous_simulation.py
"""
Simulation runner for continuous Bayesian interpersonal dynamics
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, Any, Optional
from dataclasses import dataclass

# Change these relative imports to absolute imports
from continuous_bayesian_agent import ContinuousBayesianAgent
from continuous_action_selection import select_action_continuous


@dataclass
class ContinuousSimulationResults:
    """
    Container for continuous simulation results
    """

    # Round-by-round data
    rounds: np.ndarray
    agent1_actions: np.ndarray
    agent2_actions: np.ndarray
    agent1_payoffs: np.ndarray
    agent2_payoffs: np.ndarray
    agent1_beliefs: np.ndarray  # Belief about opponent's mean action
    agent2_beliefs: np.ndarray
    agent1_uncertainty: np.ndarray  # Uncertainty in beliefs
    agent2_uncertainty: np.ndarray

    # Summary statistics
    summary: Dict[str, Any]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame for analysis"""

        return pd.DataFrame(
            {
                "round": self.rounds,
                "agent1_action": self.agent1_actions,
                "agent2_action": self.agent2_actions,
                "agent1_payoff": self.agent1_payoffs,
                "agent2_payoff": self.agent2_payoffs,
                "agent1_belief": self.agent1_beliefs,
                "agent2_belief": self.agent2_beliefs,
                "agent1_uncertainty": self.agent1_uncertainty,
                "agent2_uncertainty": self.agent2_uncertainty,
                "action_difference": np.abs(self.agent1_actions - self.agent2_actions),
                "average_action": (self.agent1_actions + self.agent2_actions) / 2,
                "total_payoff": self.agent1_payoffs + self.agent2_payoffs,
            }
        )

    def __str__(self) -> str:
        """String representation of simulation results"""
        s = self.summary
        return (
            f"Continuous Bayesian Simulation Results\n"
            f"======================================\n"
            f"Rounds: {s['total_rounds']}\n"
            f"Agent 1 - Mean action: {s['agent1_mean_action']:.3f} | "
            f"Total payoff: {s['agent1_total_payoff']:.1f}\n"
            f"Agent 2 - Mean action: {s['agent2_mean_action']:.3f} | "
            f"Total payoff: {s['agent2_total_payoff']:.1f}\n"
            f"Mean action difference: {s['mean_action_difference']:.3f}\n"
            f"Final beliefs - A1 about A2: {s['agent1_final_belief']:.3f} | "
            f"A2 about A1: {s['agent2_final_belief']:.3f}"
        )


def run_continuous_simulation(
    agent1: ContinuousBayesianAgent,
    agent2: ContinuousBayesianAgent,
    payoff_function1: Callable[[float, float], float],
    payoff_function2: Callable[[float, float], float],
    n_rounds: int = 300,
    action_selection_method: str = "thompson",
    verbose: bool = False,
    random_seed: Optional[int] = None,
) -> ContinuousSimulationResults:
    """
    Run a complete continuous simulation of two Bayesian agents.

    Args:
        agent1: First ContinuousBayesianAgent
        agent2: Second ContinuousBayesianAgent
        payoff_function1: Payoff function for agent1
        payoff_function2: Payoff function for agent2
        n_rounds: Number of interaction rounds
        action_selection_method: Method for action selection
        verbose: Whether to print progress
        random_seed: Random seed for reproducibility

    Returns:
        ContinuousSimulationResults object
    """

    # Set random seed
    if random_seed is not None:
        np.random.seed(random_seed)

    # Validate inputs
    if not isinstance(agent1, ContinuousBayesianAgent) or not isinstance(
        agent2, ContinuousBayesianAgent
    ):
        raise TypeError("Both agents must be ContinuousBayesianAgent objects")

    if action_selection_method not in ["thompson", "ucb", "softmax", "greedy"]:
        raise ValueError(
            "action_selection_method must be 'thompson', 'ucb', 'softmax', or 'greedy'"
        )

    # Initialize result storage
    results_data = {
        "rounds": np.arange(1, n_rounds + 1),
        "agent1_actions": np.zeros(n_rounds),
        "agent2_actions": np.zeros(n_rounds),
        "agent1_payoffs": np.zeros(n_rounds),
        "agent2_payoffs": np.zeros(n_rounds),
        "agent1_beliefs": np.zeros(n_rounds),
        "agent2_beliefs": np.zeros(n_rounds),
        "agent1_uncertainty": np.zeros(n_rounds),
        "agent2_uncertainty": np.zeros(n_rounds),
    }

    # Run simulation rounds
    for round_num in range(n_rounds):

        if verbose and (round_num + 1) % 50 == 0:
            print(f"Round {round_num + 1} of {n_rounds}")

        # Select actions for both agents
        action1 = select_action_continuous(
            agent1, payoff_function1, method=action_selection_method
        )
        action2 = select_action_continuous(
            agent2, payoff_function2, method=action_selection_method
        )

        # Calculate payoffs
        payoff1 = payoff_function1(action1, action2)
        payoff2 = payoff_function2(action2, action1)

        # Update beliefs based on opponent's action
        agent1.update_beliefs(np.array([action2]))
        agent2.update_beliefs(np.array([action1]))

        # Store action and payoff history in agents
        agent1.action_history.append(np.array([action1]))
        agent2.action_history.append(np.array([action2]))
        agent1.payoff_history.append(payoff1)
        agent2.payoff_history.append(payoff2)

        # Get current belief statistics
        belief_stats1 = agent1.get_belief_stats()
        belief_stats2 = agent2.get_belief_stats()

        # Store results for this round
        results_data["agent1_actions"][round_num] = action1
        results_data["agent2_actions"][round_num] = action2
        results_data["agent1_payoffs"][round_num] = payoff1
        results_data["agent2_payoffs"][round_num] = payoff2
        results_data["agent1_beliefs"][round_num] = belief_stats1["warmth_belief"]
        results_data["agent2_beliefs"][round_num] = belief_stats2["warmth_belief"]
        results_data["agent1_uncertainty"][round_num] = np.sqrt(
            belief_stats1["mean_uncertainty"][0, 0]
        )
        results_data["agent2_uncertainty"][round_num] = np.sqrt(
            belief_stats2["mean_uncertainty"][0, 0]
        )

    # Calculate summary statistics
    summary = {
        "total_rounds": n_rounds,
        "agent1_mean_action": np.mean(results_data["agent1_actions"]),
        "agent2_mean_action": np.mean(results_data["agent2_actions"]),
        "agent1_final_belief": results_data["agent1_beliefs"][-1],
        "agent2_final_belief": results_data["agent2_beliefs"][-1],
        "agent1_total_payoff": np.sum(results_data["agent1_payoffs"]),
        "agent2_total_payoff": np.sum(results_data["agent2_payoffs"]),
        "mean_action_difference": np.mean(
            np.abs(results_data["agent1_actions"] - results_data["agent2_actions"])
        ),
        "final_action_difference": abs(
            results_data["agent1_actions"][-1] - results_data["agent2_actions"][-1]
        ),
        "action_correlation": np.corrcoef(
            results_data["agent1_actions"], results_data["agent2_actions"]
        )[0, 1],
        "payoff_correlation": np.corrcoef(
            results_data["agent1_payoffs"], results_data["agent2_payoffs"]
        )[0, 1],
        "final_agents": {"agent1": agent1, "agent2": agent2},
    }

    # Create and return results object
    return ContinuousSimulationResults(
        rounds=results_data["rounds"],
        agent1_actions=results_data["agent1_actions"],
        agent2_actions=results_data["agent2_actions"],
        agent1_payoffs=results_data["agent1_payoffs"],
        agent2_payoffs=results_data["agent2_payoffs"],
        agent1_beliefs=results_data["agent1_beliefs"],
        agent2_beliefs=results_data["agent2_beliefs"],
        agent1_uncertainty=results_data["agent1_uncertainty"],
        agent2_uncertainty=results_data["agent2_uncertainty"],
        summary=summary,
    )


def run_multiple_continuous_simulations(
    agent1_params: Dict[str, Any],
    agent2_params: Dict[str, Any],
    payoff_function1: Callable[[float, float], float],
    payoff_function2: Callable[[float, float], float],
    n_simulations: int = 50,
    n_rounds: int = 300,
    action_selection_method: str = "thompson",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run multiple continuous simulations for statistical analysis.

    Args:
        agent1_params: Parameters for creating agent1
        agent2_params: Parameters for creating agent2
        payoff_function1: Payoff function for agent1
        payoff_function2: Payoff function for agent2
        n_simulations: Number of simulation runs
        n_rounds: Number of rounds per simulation
        action_selection_method: Action selection method
        verbose: Whether to print progress

    Returns:
        DataFrame with summary statistics from all simulations
    """

    from continuous_bayesian_agent import create_continuous_bayesian_agent

    results_list = []

    for sim_num in range(n_simulations):

        if verbose and (sim_num + 1) % 10 == 0:
            print(f"Simulation {sim_num + 1} of {n_simulations}")

        # Create fresh agents for each simulation
        agent1 = create_continuous_bayesian_agent(**agent1_params)
        agent2 = create_continuous_bayesian_agent(**agent2_params)

        # Run simulation
        results = run_continuous_simulation(
            agent1=agent1,
            agent2=agent2,
            payoff_function1=payoff_function1,
            payoff_function2=payoff_function2,
            n_rounds=n_rounds,
            action_selection_method=action_selection_method,
            random_seed=sim_num,  # Different seed for each simulation
        )

        # Extract summary statistics
        summary_row = {
            "simulation": sim_num + 1,
            "agent1_mean_action": results.summary["agent1_mean_action"],
            "agent2_mean_action": results.summary["agent2_mean_action"],
            "agent1_total_payoff": results.summary["agent1_total_payoff"],
            "agent2_total_payoff": results.summary["agent2_total_payoff"],
            "mean_action_difference": results.summary["mean_action_difference"],
            "final_action_difference": results.summary["final_action_difference"],
            "action_correlation": results.summary["action_correlation"],
            "agent1_final_belief": results.summary["agent1_final_belief"],
            "agent2_final_belief": results.summary["agent2_final_belief"],
        }

        results_list.append(summary_row)

    return pd.DataFrame(results_list)


def analyze_convergence_continuous(
    results: ContinuousSimulationResults, window_size: int = 50
) -> Dict[str, Any]:
    """
    Analyze convergence patterns in continuous simulation.

    Args:
        results: ContinuousSimulationResults object
        window_size: Window size for stability analysis

    Returns:
        Dictionary with convergence metrics
    """

    n_rounds = len(results.rounds)

    if window_size >= n_rounds:
        raise ValueError("window_size must be smaller than total rounds")

    # Final window for stability analysis
    final_window_start = n_rounds - window_size
    final_actions_1 = results.agent1_actions[final_window_start:]
    final_actions_2 = results.agent2_actions[final_window_start:]
    final_beliefs_1 = results.agent1_beliefs[final_window_start:]
    final_beliefs_2 = results.agent2_beliefs[final_window_start:]

    # Calculate stability metrics
    action_stability_1 = 1 / (1 + np.var(final_actions_1))
    action_stability_2 = 1 / (1 + np.var(final_actions_2))
    belief_stability_1 = 1 / (1 + np.var(final_beliefs_1))
    belief_stability_2 = 1 / (1 + np.var(final_beliefs_2))

    # Belief accuracy
    final_mean_action_1 = np.mean(final_actions_1)
    final_mean_action_2 = np.mean(final_actions_2)
    belief_accuracy_1 = 1 - abs(
        results.summary["agent1_final_belief"] - final_mean_action_2
    )
    belief_accuracy_2 = 1 - abs(
        results.summary["agent2_final_belief"] - final_mean_action_1
    )

    # Action matching (how close agents' actions are)
    final_action_matching = 1 - np.mean(np.abs(final_actions_1 - final_actions_2))

    # Overall convergence score
    convergence_score = np.mean(
        [
            action_stability_1,
            action_stability_2,
            belief_stability_1,
            belief_stability_2,
            max(0, belief_accuracy_1),
            max(0, belief_accuracy_2),
            max(0, final_action_matching),
        ]
    )

    return {
        "action_stability": {
            "agent1": action_stability_1,
            "agent2": action_stability_2,
        },
        "belief_stability": {
            "agent1": belief_stability_1,
            "agent2": belief_stability_2,
        },
        "belief_accuracy": {"agent1": belief_accuracy_1, "agent2": belief_accuracy_2},
        "final_action_matching": final_action_matching,
        "convergence_score": convergence_score,
        "window_size": window_size,
    }
