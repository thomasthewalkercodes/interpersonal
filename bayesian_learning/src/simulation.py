# src/simulation.py
"""
Main simulation runner for Bayesian interpersonal dynamics
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Literal, Tuple
from dataclasses import dataclass
from .bayesian_agent import BayesianAgent
from .belief_update import update_beliefs, calculate_belief_uncertainty
from .action_selection import select_action


@dataclass
class SimulationResults:
    """
    Container for simulation results with convenient access methods
    """

    # Round-by-round data
    rounds: np.ndarray
    agent1_actions: np.ndarray
    agent2_actions: np.ndarray
    agent1_payoffs: np.ndarray
    agent2_payoffs: np.ndarray
    agent1_beliefs: np.ndarray
    agent2_beliefs: np.ndarray
    agent1_uncertainty: np.ndarray
    agent2_uncertainty: np.ndarray

    # Summary statistics
    summary: Dict[str, Any]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame for easy analysis"""

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
                "mutual_cooperation": (
                    (self.agent1_actions == 1) & (self.agent2_actions == 1)
                ).astype(int),
            }
        )

    def __str__(self) -> str:
        """String representation of simulation results"""
        s = self.summary
        return (
            f"Bayesian Interpersonal Simulation Results\n"
            f"=========================================\n"
            f"Rounds: {s['total_rounds']}\n"
            f"Agent 1 - Warm rate: {s['agent1_warm_rate']:.3f} | "
            f"Total payoff: {s['agent1_total_payoff']:.1f}\n"
            f"Agent 2 - Warm rate: {s['agent2_warm_rate']:.3f} | "
            f"Total payoff: {s['agent2_total_payoff']:.1f}\n"
            f"Cooperation rate (both warm): {s['cooperation_rate']:.3f}\n"
            f"Final beliefs - A1 about A2: {s['agent1_final_belief']:.3f} | "
            f"A2 about A1: {s['agent2_final_belief']:.3f}"
        )


def run_bayesian_simulation(
    agent1: BayesianAgent,
    agent2: BayesianAgent,
    payoff_matrix1: np.ndarray,
    payoff_matrix2: np.ndarray,
    n_rounds: int = 300,
    action_selection_method: Literal["thompson", "ucb", "softmax"] = "thompson",
    verbose: bool = False,
    random_seed: int = None,
) -> SimulationResults:
    """
    Runs a complete simulation of two Bayesian agents interacting
    in an interpersonal dynamics game (2x2 matrix).

    Args:
        agent1: BayesianAgent object, first agent
        agent2: BayesianAgent object, second agent
        payoff_matrix1: 2x2 numpy array, payoff structure for agent1
        payoff_matrix2: 2x2 numpy array, payoff structure for agent2
        n_rounds: Number of interaction rounds
        action_selection_method: Method for action selection ("thompson", "ucb", "softmax")
        verbose: Whether to print progress
        random_seed: Random seed for reproducibility

    Returns:
        SimulationResults object containing all simulation data

    Raises:
        TypeError: If agents are not BayesianAgent objects
        ValueError: If action_selection_method is invalid
        ValueError: If payoff matrices have wrong shape
    """

    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    # Validate inputs
    if not isinstance(agent1, BayesianAgent) or not isinstance(agent2, BayesianAgent):
        raise TypeError("Both agents must be BayesianAgent objects")

    if action_selection_method not in ["thompson", "ucb", "softmax"]:
        raise ValueError(
            "action_selection_method must be 'thompson', 'ucb', or 'softmax'"
        )

    if (
        not isinstance(payoff_matrix1, np.ndarray)
        or payoff_matrix1.shape != (2, 2)
        or not isinstance(payoff_matrix2, np.ndarray)
        or payoff_matrix2.shape != (2, 2)
    ):
        raise ValueError("Both payoff matrices must be 2x2 numpy arrays")

    # Initialize result storage
    results_data = {
        "rounds": np.arange(1, n_rounds + 1),
        "agent1_actions": np.zeros(n_rounds, dtype=int),
        "agent2_actions": np.zeros(n_rounds, dtype=int),
        "agent1_payoffs": np.zeros(n_rounds, dtype=float),
        "agent2_payoffs": np.zeros(n_rounds, dtype=float),
        "agent1_beliefs": np.zeros(n_rounds, dtype=float),
        "agent2_beliefs": np.zeros(n_rounds, dtype=float),
        "agent1_uncertainty": np.zeros(n_rounds, dtype=float),
        "agent2_uncertainty": np.zeros(n_rounds, dtype=float),
    }

    # Run simulation rounds
    for round_num in range(n_rounds):

        if verbose and (round_num + 1) % 50 == 0:
            print(f"Round {round_num + 1} of {n_rounds}")

        # Select actions for both agents
        action1 = select_action(agent1, payoff_matrix1, method=action_selection_method)
        action2 = select_action(agent2, payoff_matrix2, method=action_selection_method)

        # Calculate payoffs
        # Payoff matrix indexing: [own_action, opponent_action]
        payoff1 = payoff_matrix1[action1, action2]
        payoff2 = payoff_matrix2[action2, action1]

        # Update beliefs based on opponent's action
        agent1 = update_beliefs(agent1, action2)
        agent2 = update_beliefs(agent2, action1)

        # Store action and payoff history in agents
        agent1.action_history.append(action1)
        agent2.action_history.append(action2)
        agent1.payoff_history.append(payoff1)
        agent2.payoff_history.append(payoff2)

        # Store results for this round
        results_data["agent1_actions"][round_num] = action1
        results_data["agent2_actions"][round_num] = action2
        results_data["agent1_payoffs"][round_num] = payoff1
        results_data["agent2_payoffs"][round_num] = payoff2
        results_data["agent1_beliefs"][round_num] = agent1.opponent_warm_prob
        results_data["agent2_beliefs"][round_num] = agent2.opponent_warm_prob
        results_data["agent1_uncertainty"][round_num] = calculate_belief_uncertainty(
            agent1
        )
        results_data["agent2_uncertainty"][round_num] = calculate_belief_uncertainty(
            agent2
        )

    # Calculate summary statistics
    summary = {
        "total_rounds": n_rounds,
        "agent1_final_belief": agent1.opponent_warm_prob,
        "agent2_final_belief": agent2.opponent_warm_prob,
        "agent1_total_payoff": np.sum(agent1.payoff_history),
        "agent2_total_payoff": np.sum(agent2.payoff_history),
        "agent1_warm_rate": np.mean(results_data["agent1_actions"]),
        "agent2_warm_rate": np.mean(results_data["agent2_actions"]),
        "cooperation_rate": np.mean(
            (results_data["agent1_actions"] == 1)
            & (results_data["agent2_actions"] == 1)
        ),
        "final_agents": {"agent1": agent1, "agent2": agent2},
    }

    # Create and return SimulationResults object
    return SimulationResults(
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


def run_multiple_simulations(
    agent1_params: Dict[str, Any],
    agent2_params: Dict[str, Any],
    payoff_matrix: np.ndarray,
    n_simulations: int = 50,
    n_rounds: int = 300,
    action_selection_method: str = "thompson",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Run multiple simulations with the same parameters for statistical analysis.

    Args:
        agent1_params: Dictionary of parameters for agent 1
        agent2_params: Dictionary of parameters for agent 2
        payoff_matrix: 2x2 numpy array, payoff structure (same for both agents)
        n_simulations: Number of simulation runs
        n_rounds: Number of rounds per simulation
        action_selection_method: Action selection method
        verbose: Whether to print progress

    Returns:
        DataFrame with summary statistics from all simulations
    """

    from .bayesian_agent import create_bayesian_agent

    results_list = []

    for sim_num in range(n_simulations):

        if verbose and (sim_num + 1) % 10 == 0:
            print(f"Simulation {sim_num + 1} of {n_simulations}")

        # Create fresh agents for each simulation
        agent1 = create_bayesian_agent(**agent1_params)
        agent2 = create_bayesian_agent(**agent2_params)

        # Run simulation
        results = run_bayesian_simulation(
            agent1=agent1,
            agent2=agent2,
            payoff_matrix1=payoff_matrix,
            payoff_matrix2=payoff_matrix,
            n_rounds=n_rounds,
            action_selection_method=action_selection_method,
            random_seed=sim_num,  # Different seed for each simulation
        )

        # Extract summary statistics
        summary_row = {
            "simulation": sim_num + 1,
            "cooperation_rate": results.summary["cooperation_rate"],
            "agent1_warm_rate": results.summary["agent1_warm_rate"],
            "agent2_warm_rate": results.summary["agent2_warm_rate"],
            "agent1_total_payoff": results.summary["agent1_total_payoff"],
            "agent2_total_payoff": results.summary["agent2_total_payoff"],
            "agent1_final_belief": results.summary["agent1_final_belief"],
            "agent2_final_belief": results.summary["agent2_final_belief"],
        }

        results_list.append(summary_row)

    return pd.DataFrame(results_list)
