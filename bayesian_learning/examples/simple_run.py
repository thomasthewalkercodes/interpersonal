# examples/simple_run.py
"""
Simple example demonstrating Bayesian interpersonal dynamics
"""

import sys
import os
import numpy as np

from ..src.continuous_bayesian_agent import create_bayesian_agent
from ..src.continuous_simulation import run_bayesian_simulation
from ..config.bayesian_config import (
    create_default_config,
    create_agent_preset,
)


def run_simple_example():
    """
    Demonstrates basic usage of the Bayesian interpersonal dynamics model
    """

    print("Running Simple Bayesian Interpersonal Dynamics Example")
    print("=" * 55)
    print()

    # Load default configuration
    config = create_default_config()

    # Create two agents with different personality profiles
    # Agent 1: Secure attachment (optimistic about others)
    agent1_params = create_agent_preset("secure")
    agent1 = create_bayesian_agent(agent_id="Secure_Agent", **agent1_params)

    # Agent 2: Anxious attachment (pessimistic, loss averse)
    agent2_params = create_agent_preset("anxious")
    agent2 = create_bayesian_agent(agent_id="Anxious_Agent", **agent2_params)

    print("Initial Agent States:")
    print(agent1)
    print()
    print(agent2)
    print()

    # Set up payoff matrices (same for both agents in this example)
    payoff_matrix = config["payoffs"]["default_matrix"]
    print("Payoff Matrix (rows=own actions, cols=opponent actions):")
    print("       Cold  Warm")
    print(f"Cold:    {payoff_matrix[0,0]}     {payoff_matrix[0,1]}")
    print(f"Warm:    {payoff_matrix[1,0]}     {payoff_matrix[1,1]}")
    print()

    # Run simulation
    print(f"Running simulation for {config['simulation']['n_rounds']} rounds...")
    results = run_bayesian_simulation(
        agent1=agent1,
        agent2=agent2,
        payoff_matrix1=payoff_matrix,
        payoff_matrix2=payoff_matrix,
        n_rounds=config["simulation"]["n_rounds"],
        action_selection_method=config["simulation"]["action_selection_method"],
        verbose=True,
    )

    # Display results
    print()
    print(results)

    # Show final agent states
    print("\nFinal Agent States:")
    print(results.summary["final_agents"]["agent1"])
    print()
    print(results.summary["final_agents"]["agent2"])

    # Basic analysis
    print("\nBasic Analysis:")
    print("=" * 15)
    convergence = "YES" if results.summary["cooperation_rate"] > 0.7 else "NO"
    print(f"Convergence to mutual cooperation: {convergence}")

    belief_accuracy_1 = abs(
        results.summary["agent1_final_belief"] - results.summary["agent2_warm_rate"]
    )
    belief_accuracy_2 = abs(
        results.summary["agent2_final_belief"] - results.summary["agent1_warm_rate"]
    )

    print(f"Belief accuracy - Agent1: {belief_accuracy_1:.3f} (lower is better)")
    print(f"Belief accuracy - Agent2: {belief_accuracy_2:.3f} (lower is better)")

    # Convert to DataFrame for further analysis
    df = results.to_dataframe()
    print(f"\nDataFrame shape: {df.shape}")
    print("First 5 rounds:")
    print(df.head())

    return results


def run_comparison_example():
    """
    Demonstrates comparing different personality combinations
    """

    print("\n" + "=" * 60)
    print("Comparison Example: Different Personality Combinations")
    print("=" * 60)

    from continuous_simulation import run_multiple_simulations

    # Define personality types to compare
    personality_types = ["secure", "anxious", "avoidant"]
    payoff_matrix = create_default_config()["payoffs"]["default_matrix"]

    results_summary = []

    for p1 in personality_types:
        for p2 in personality_types:

            print(f"\nRunning {p1} vs {p2}...")

            # Get agent parameters
            agent1_params = create_agent_preset(p1)
            agent2_params = create_agent_preset(p2)

            # Add agent_id to parameters
            agent1_params["agent_id"] = f"{p1}_agent"
            agent2_params["agent_id"] = f"{p2}_agent"

            # Run multiple simulations
            multi_results = run_multiple_simulations(
                agent1_params=agent1_params,
                agent2_params=agent2_params,
                payoff_matrix=payoff_matrix,
                n_simulations=10,  # Reduced for demo
                n_rounds=300,
                verbose=False,
            )

            # Calculate summary statistics
            summary = {
                "agent1_type": p1,
                "agent2_type": p2,
                "mean_cooperation": multi_results["cooperation_rate"].mean(),
                "std_cooperation": multi_results["cooperation_rate"].std(),
                "mean_payoff_diff": (
                    multi_results["agent1_total_payoff"]
                    - multi_results["agent2_total_payoff"]
                ).mean(),
            }

            results_summary.append(summary)

            print(
                f"  Mean cooperation rate: {summary['mean_cooperation']:.3f} ± {summary['std_cooperation']:.3f}"
            )
            print(f"  Mean payoff difference: {summary['mean_payoff_diff']:.1f}")

    # Display comparison table
    import pandas as pd

    comparison_df = pd.DataFrame(results_summary)
    print("\nFull Comparison Results:")
    print(comparison_df.round(3))

    return comparison_df


if __name__ == "__main__":
    # Run basic example
    results = run_simple_example()

    # Run comparison example
    comparison_results = run_comparison_example()

    print("\nExample completed successfully!")
