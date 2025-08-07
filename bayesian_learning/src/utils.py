# src/utils.py
"""
Utility functions for Bayesian interpersonal dynamics simulation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
from .continuous_simulation import SimulationResults


def plot_simulation_results(
    results: SimulationResults,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
) -> plt.Figure:
    """
    Creates visualization of simulation results showing belief evolution,
    actions, and payoffs over time

    Args:
        results: SimulationResults object
        save_path: Optional path to save plot
        figsize: Figure size as (width, height)

    Returns:
        matplotlib Figure object
    """

    # Set style
    plt.style.use("seaborn-v0_8")
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    rounds = results.rounds

    # Panel 1: Belief evolution
    axes[0].plot(
        rounds, results.agent1_beliefs, label="Agent 1", linewidth=2, alpha=0.8
    )
    axes[0].plot(
        rounds, results.agent2_beliefs, label="Agent 2", linewidth=2, alpha=0.8
    )
    axes[0].set_ylabel("Belief about opponent warmth")
    axes[0].set_title("Belief Evolution Over Time", fontsize=14, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)

    # Panel 2: Actions over time (smoothed)
    window_size = min(50, len(rounds) // 10)
    if window_size > 1:
        agent1_smooth = (
            pd.Series(results.agent1_actions).rolling(window_size, center=True).mean()
        )
        agent2_smooth = (
            pd.Series(results.agent2_actions).rolling(window_size, center=True).mean()
        )
    else:
        agent1_smooth = results.agent1_actions
        agent2_smooth = results.agent2_actions

    axes[1].plot(rounds, agent1_smooth, label="Agent 1", linewidth=2, alpha=0.8)
    axes[1].plot(rounds, agent2_smooth, label="Agent 2", linewidth=2, alpha=0.8)
    axes[1].set_ylabel("Warmth rate (smoothed)")
    axes[1].set_title("Action Patterns Over Time", fontsize=14, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    # Panel 3: Uncertainty over time
    axes[2].plot(
        rounds, results.agent1_uncertainty, label="Agent 1", linewidth=2, alpha=0.8
    )
    axes[2].plot(
        rounds, results.agent2_uncertainty, label="Agent 2", linewidth=2, alpha=0.8
    )
    axes[2].set_ylabel("Belief uncertainty")
    axes[2].set_xlabel("Round")
    axes[2].set_title("Belief Uncertainty Over Time", fontsize=14, fontweight="bold")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    return fig


def calculate_convergence_metrics(
    results: SimulationResults, window_size: int = 50
) -> Dict[str, Any]:
    """
    Calculates metrics to assess whether agents converged to stable patterns

    Args:
        results: SimulationResults object
        window_size: Size of window for stability assessment

    Returns:
        Dictionary containing convergence metrics

    Raises:
        ValueError: If window_size is too large
    """

    n_rounds = len(results.rounds)

    if window_size >= n_rounds:
        raise ValueError("window_size must be smaller than total rounds")

    # Calculate stability in final window
    final_window_start = n_rounds - window_size
    final_beliefs_1 = results.agent1_beliefs[final_window_start:]
    final_beliefs_2 = results.agent2_beliefs[final_window_start:]
    final_actions_1 = results.agent1_actions[final_window_start:]
    final_actions_2 = results.agent2_actions[final_window_start:]

    # Belief stability (low variance in final window)
    agent1_belief_stability = np.var(final_beliefs_1)
    agent2_belief_stability = np.var(final_beliefs_2)

    # Action stability (consistency in final window)
    agent1_action_consistency = 1 - np.var(final_actions_1)
    agent2_action_consistency = 1 - np.var(final_actions_2)

    # Mutual cooperation rate in final window
    final_cooperation_rate = np.mean((final_actions_1 == 1) & (final_actions_2 == 1))

    # Belief accuracy (how well final beliefs predict actual behavior)
    agent1_belief_accuracy = 1 - abs(
        results.summary["agent1_final_belief"] - results.summary["agent2_warm_rate"]
    )
    agent2_belief_accuracy = 1 - abs(
        results.summary["agent2_final_belief"] - results.summary["agent1_warm_rate"]
    )

    # Overall convergence metric
    overall_convergence = np.mean(
        [
            1 - agent1_belief_stability,
            1 - agent2_belief_stability,
            max(0, agent1_belief_accuracy),
            max(0, agent2_belief_accuracy),
        ]
    )

    metrics = {
        "belief_stability": {
            "agent1": agent1_belief_stability,
            "agent2": agent2_belief_stability,
        },
        "action_consistency": {
            "agent1": max(0, agent1_action_consistency),
            "agent2": max(0, agent2_action_consistency),
        },
        "final_cooperation_rate": final_cooperation_rate,
        "belief_accuracy": {
            "agent1": agent1_belief_accuracy,
            "agent2": agent2_belief_accuracy,
        },
        "overall_convergence": overall_convergence,
        "window_size": window_size,
    }

    return metrics


def create_heatmap_analysis(
    results_df: pd.DataFrame,
    metric: str = "cooperation_rate",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Creates a heatmap showing how different agent combinations perform

    Args:
        results_df: DataFrame with columns including agent types and metrics
        metric: Column name to use for the heatmap values
        save_path: Optional path to save plot

    Returns:
        matplotlib Figure object
    """

    # Pivot the DataFrame to create a matrix
    if "agent1_type" in results_df.columns and "agent2_type" in results_df.columns:
        pivot_df = results_df.pivot_table(
            values=metric, index="agent1_type", columns="agent2_type", aggfunc="mean"
        )
    else:
        raise ValueError(
            "DataFrame must contain 'agent1_type' and 'agent2_type' columns"
        )

    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        pivot_df,
        annot=True,
        cmap="viridis",
        center=pivot_df.values.mean(),
        square=True,
        cbar_kws={"label": metric},
        ax=ax,
    )

    ax.set_title(
        f"Agent Combination Performance: {metric}", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Agent 2 Type")
    ax.set_ylabel("Agent 1 Type")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Heatmap saved to: {save_path}")

    return fig


def analyze_belief_evolution(
    results: SimulationResults, phases: List[Tuple[int, int]] = None
) -> Dict[str, Any]:
    """
    Analyzes how beliefs evolve over different phases of the simulation

    Args:
        results: SimulationResults object
        phases: List of (start, end) tuples defining phases to analyze
                If None, uses early/middle/late phases

    Returns:
        Dictionary with analysis results for each phase
    """

    n_rounds = len(results.rounds)

    if phases is None:
        # Default phases: early, middle, late
        phase_size = n_rounds // 3
        phases = [
            (0, phase_size),
            (phase_size, 2 * phase_size),
            (2 * phase_size, n_rounds),
        ]

    phase_names = [f"Phase_{i+1}" for i in range(len(phases))]
    analysis = {}

    for phase_name, (start, end) in zip(phase_names, phases):

        # Extract data for this phase
        phase_beliefs_1 = results.agent1_beliefs[start:end]
        phase_beliefs_2 = results.agent2_beliefs[start:end]
        phase_actions_1 = results.agent1_actions[start:end]
        phase_actions_2 = results.agent2_actions[start:end]
        phase_uncertainty_1 = results.agent1_uncertainty[start:end]
        phase_uncertainty_2 = results.agent2_uncertainty[start:end]

        analysis[phase_name] = {
            "rounds": (start, end),
            "agent1_mean_belief": np.mean(phase_beliefs_1),
            "agent2_mean_belief": np.mean(phase_beliefs_2),
            "agent1_belief_stability": 1 / (1 + np.var(phase_beliefs_1)),
            "agent2_belief_stability": 1 / (1 + np.var(phase_beliefs_2)),
            "agent1_warmth_rate": np.mean(phase_actions_1),
            "agent2_warmth_rate": np.mean(phase_actions_2),
            "cooperation_rate": np.mean(
                (phase_actions_1 == 1) & (phase_actions_2 == 1)
            ),
            "agent1_mean_uncertainty": np.mean(phase_uncertainty_1),
            "agent2_mean_uncertainty": np.mean(phase_uncertainty_2),
        }

    return analysis


def export_results_to_csv(results: SimulationResults, filename: str) -> None:
    """
    Exports simulation results to CSV format for external analysis

    Args:
        results: SimulationResults object
        filename: Output filename (should end with .csv)
    """

    df = results.to_dataframe()
    df.to_csv(filename, index=False)
    print(f"Results exported to: {filename}")


def create_summary_statistics(results_list: List[SimulationResults]) -> pd.DataFrame:
    """
    Creates summary statistics from multiple simulation results

    Args:
        results_list: List of SimulationResults objects

    Returns:
        DataFrame with summary statistics across all simulations
    """

    summaries = []

    for i, results in enumerate(results_list):
        summary = results.summary.copy()
        summary["simulation_id"] = i

        # Add convergence metrics
        convergence = calculate_convergence_metrics(results)
        summary["overall_convergence"] = convergence["overall_convergence"]
        summary["final_cooperation_rate"] = convergence["final_cooperation_rate"]

        summaries.append(summary)

    df = pd.DataFrame(summaries)

    # Calculate aggregate statistics
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    aggregate_stats = {}
    for col in numeric_columns:
        if col != "simulation_id":
            aggregate_stats[f"{col}_mean"] = df[col].mean()
            aggregate_stats[f"{col}_std"] = df[col].std()
            aggregate_stats[f"{col}_min"] = df[col].min()
            aggregate_stats[f"{col}_max"] = df[col].max()

    # Add aggregate row
    agg_row = pd.Series(aggregate_stats)
    agg_df = pd.DataFrame([agg_row])
    agg_df["simulation_id"] = "AGGREGATE"

    return pd.concat([df, agg_df], ignore_index=True)


def plot_belief_distributions(
    results: SimulationResults,
    rounds_to_plot: List[int] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plots the evolution of belief distributions over selected rounds

    Args:
        results: SimulationResults object
        rounds_to_plot: List of round numbers to plot. If None, uses evenly spaced rounds
        save_path: Optional path to save plot

    Returns:
        matplotlib Figure object
    """

    from scipy.stats import beta

    if rounds_to_plot is None:
        n_plots = min(6, len(results.rounds) // 50)  # Plot every 50 rounds, max 6 plots
        rounds_to_plot = np.linspace(0, len(results.rounds) - 1, n_plots, dtype=int)

    fig, axes = plt.subplots(
        2, len(rounds_to_plot), figsize=(3 * len(rounds_to_plot), 6)
    )
    if len(rounds_to_plot) == 1:
        axes = axes.reshape(-1, 1)

    x = np.linspace(0, 1, 100)

    # Get final agents to reconstruct belief evolution
    final_agents = results.summary["final_agents"]
    agent1 = final_agents["agent1"]
    agent2 = final_agents["agent2"]

    for i, round_idx in enumerate(rounds_to_plot):

        # Reconstruct belief parameters at this round
        # This is approximate - we estimate alpha/beta from stored beliefs and histories
        round_num = round_idx + 1

        # Agent 1 beliefs about Agent 2
        if round_idx < len(agent1.opponent_history):
            warm_count = sum(agent1.opponent_history[: round_idx + 1])
            cold_count = round_idx + 1 - warm_count
            alpha1 = 1.0 + warm_count  # Assuming initial priors were 1.0
            beta1 = 1.0 + cold_count
        else:
            alpha1, beta1 = 1.0, 1.0

        # Agent 2 beliefs about Agent 1
        if round_idx < len(agent2.opponent_history):
            warm_count = sum(agent2.opponent_history[: round_idx + 1])
            cold_count = round_idx + 1 - warm_count
            alpha2 = 1.0 + warm_count
            beta2 = 1.0 + cold_count
        else:
            alpha2, beta2 = 1.0, 1.0

        # Plot Agent 1's belief distribution
        y1 = beta.pdf(x, alpha1, beta1)
        axes[0, i].plot(
            x, y1, "b-", linewidth=2, label=f"Beta({alpha1:.1f}, {beta1:.1f})"
        )
        axes[0, i].axvline(
            alpha1 / (alpha1 + beta1), color="b", linestyle="--", alpha=0.7
        )
        axes[0, i].set_title(f"Agent 1\nRound {round_num}")
        axes[0, i].set_xlabel("Opponent warmth probability")
        axes[0, i].set_ylabel("Density")
        axes[0, i].legend(fontsize="small")
        axes[0, i].grid(True, alpha=0.3)

        # Plot Agent 2's belief distribution
        y2 = beta.pdf(x, alpha2, beta2)
        axes[1, i].plot(
            x, y2, "r-", linewidth=2, label=f"Beta({alpha2:.1f}, {beta2:.1f})"
        )
        axes[1, i].axvline(
            alpha2 / (alpha2 + beta2), color="r", linestyle="--", alpha=0.7
        )
        axes[1, i].set_title(f"Agent 2\nRound {round_num}")
        axes[1, i].set_xlabel("Opponent warmth probability")
        axes[1, i].set_ylabel("Density")
        axes[1, i].legend(fontsize="small")
        axes[1, i].grid(True, alpha=0.3)

    plt.suptitle("Evolution of Belief Distributions", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Belief distribution plot saved to: {save_path}")

    return fig


def create_interaction_matrix_plot(
    results: SimulationResults, save_path: Optional[str] = None
) -> plt.Figure:
    """
    Creates a plot showing the frequency of different action combinations

    Args:
        results: SimulationResults object
        save_path: Optional path to save plot

    Returns:
        matplotlib Figure object
    """

    # Count action combinations
    action_combinations = np.zeros((2, 2))

    for a1, a2 in zip(results.agent1_actions, results.agent2_actions):
        action_combinations[a1, a2] += 1

    # Convert to percentages
    action_percentages = action_combinations / len(results.rounds) * 100

    # Create labels
    labels = [["Cold-Cold", "Cold-Warm"], ["Warm-Cold", "Warm-Warm"]]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(action_percentages, cmap="Blues", aspect="equal")

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(
                j,
                i,
                f"{labels[i][j]}\n{action_percentages[i, j]:.1f}%",
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=12,
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Agent 2: Cold", "Agent 2: Warm"])
    ax.set_yticklabels(["Agent 1: Cold", "Agent 1: Warm"])
    ax.set_title(
        "Interaction Patterns (% of total rounds)", fontsize=14, fontweight="bold"
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Percentage of rounds", rotation=270, labelpad=15)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Interaction matrix plot saved to: {save_path}")

    return fig


def compare_action_selection_methods(
    agent1_params: Dict[str, Any],
    agent2_params: Dict[str, Any],
    payoff_matrix: np.ndarray,
    n_simulations: int = 20,
    n_rounds: int = 300,
) -> pd.DataFrame:
    """
    Compares different action selection methods for the same agent configuration

    Args:
        agent1_params: Parameters for agent 1
        agent2_params: Parameters for agent 2
        payoff_matrix: Payoff matrix for both agents
        n_simulations: Number of simulations per method
        n_rounds: Number of rounds per simulation

    Returns:
        DataFrame comparing the performance of different methods
    """

    from .continuous_simulation import run_multiple_simulations

    methods = ["thompson", "ucb", "softmax"]
    comparison_results = []

    for method in methods:
        print(f"Testing {method} method...")

        # Run multiple simulations with this method
        method_results = run_multiple_simulations(
            agent1_params=agent1_params,
            agent2_params=agent2_params,
            payoff_matrix=payoff_matrix,
            n_simulations=n_simulations,
            n_rounds=n_rounds,
            action_selection_method=method,
            verbose=False,
        )

        # Calculate summary statistics
        summary = {
            "method": method,
            "mean_cooperation": method_results["cooperation_rate"].mean(),
            "std_cooperation": method_results["cooperation_rate"].std(),
            "mean_agent1_payoff": method_results["agent1_total_payoff"].mean(),
            "mean_agent2_payoff": method_results["agent2_total_payoff"].mean(),
            "mean_total_payoff": (
                method_results["agent1_total_payoff"]
                + method_results["agent2_total_payoff"]
            ).mean(),
            "cooperation_stability": 1 - method_results["cooperation_rate"].std(),
        }

        comparison_results.append(summary)

    return pd.DataFrame(comparison_results)


def create_comprehensive_report(
    results: SimulationResults, output_dir: str = "reports"
) -> Dict[str, str]:
    """
    Creates a comprehensive analysis report with multiple visualizations

    Args:
        results: SimulationResults object
        output_dir: Directory to save report files

    Returns:
        Dictionary mapping report components to their file paths
    """

    import os

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    report_files = {}

    # 1. Main simulation plot
    main_plot_path = os.path.join(output_dir, "simulation_overview.png")
    plot_simulation_results(results, save_path=main_plot_path)
    report_files["main_plot"] = main_plot_path

    # 2. Interaction matrix
    matrix_plot_path = os.path.join(output_dir, "interaction_matrix.png")
    create_interaction_matrix_plot(results, save_path=matrix_plot_path)
    report_files["interaction_matrix"] = matrix_plot_path

    # 3. Belief distributions
    belief_plot_path = os.path.join(output_dir, "belief_evolution.png")
    plot_belief_distributions(results, save_path=belief_plot_path)
    report_files["belief_distributions"] = belief_plot_path

    # 4. Raw data export
    data_path = os.path.join(output_dir, "simulation_data.csv")
    export_results_to_csv(results, data_path)
    report_files["raw_data"] = data_path

    # 5. Summary statistics
    summary_path = os.path.join(output_dir, "summary_report.txt")
    with open(summary_path, "w") as f:
        f.write("BAYESIAN INTERPERSONAL DYNAMICS SIMULATION REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write(str(results) + "\n\n")

        # Convergence analysis
        convergence = calculate_convergence_metrics(results)
        f.write("CONVERGENCE ANALYSIS:\n")
        f.write(
            f"Overall convergence score: {convergence['overall_convergence']:.3f}\n"
        )
        f.write(
            f"Final cooperation rate: {convergence['final_cooperation_rate']:.3f}\n"
        )
        f.write(
            f"Belief stability - Agent 1: {convergence['belief_stability']['agent1']:.4f}\n"
        )
        f.write(
            f"Belief stability - Agent 2: {convergence['belief_stability']['agent2']:.4f}\n"
        )
        f.write(
            f"Belief accuracy - Agent 1: {convergence['belief_accuracy']['agent1']:.3f}\n"
        )
        f.write(
            f"Belief accuracy - Agent 2: {convergence['belief_accuracy']['agent2']:.3f}\n\n"
        )

        # Phase analysis
        phase_analysis = analyze_belief_evolution(results)
        f.write("PHASE ANALYSIS:\n")
        for phase_name, phase_data in phase_analysis.items():
            f.write(
                f"{phase_name} (rounds {phase_data['rounds'][0]}-{phase_data['rounds'][1]}):\n"
            )
            f.write(f"  Cooperation rate: {phase_data['cooperation_rate']:.3f}\n")
            f.write(f"  Agent 1 mean belief: {phase_data['agent1_mean_belief']:.3f}\n")
            f.write(f"  Agent 2 mean belief: {phase_data['agent2_mean_belief']:.3f}\n")
            f.write(
                f"  Agent 1 uncertainty: {phase_data['agent1_mean_uncertainty']:.4f}\n"
            )
            f.write(
                f"  Agent 2 uncertainty: {phase_data['agent2_mean_uncertainty']:.4f}\n\n"
            )

    report_files["summary_report"] = summary_path

    print(f"Comprehensive report generated in: {output_dir}")
    print(f"Files created: {list(report_files.keys())}")

    return report_files
