"""
Example usage of the SAC Control Center for interpersonal agent simulations.

This script demonstrates various ways to use the control system:
1. Simple pairwise interactions
2. Custom agent configurations
3. Comparison studies
4. Parameter sweeps
"""

import os
import sys
from typing import List, Dict, Any

# Add the project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from control_center.sac_control import SACSControlCenter, SimulationConfig
from sim_plots.sac_plot import SimulationPlotter


def example_1_basic_interaction():
    """Example 1: Basic interaction between two different agent types."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Cooperative vs Competitive Interaction")
    print("=" * 60)

    control = SACSControlCenter()

    # Simple configuration
    config = SimulationConfig(
        agent1_type="cooperative",
        agent2_type="competitive",
        episodes=500,
        steps_per_episode=30,
        save_plots=True,
        run_name="example_1_coop_vs_comp",
    )

    results = control.run_simulation(config)

    # Print summary
    final_eval = results["final_evaluation"]
    print(f"\nFinal Results:")
    print(f"Cooperative agent average reward: {final_eval['agent1_avg_reward']:.3f}")
    print(f"Competitive agent average reward: {final_eval['agent2_avg_reward']:.3f}")

    return results


def example_2_custom_agents():
    """Example 2: Custom agent configurations with specific parameters."""
    print("=" * 60)
    print("EXAMPLE 2: Custom Agent Configurations")
    print("=" * 60)

    control = SACSControlCenter()

    # Create a trusting agent vs a suspicious agent
    config = SimulationConfig(
        agent1_type="base",
        agent2_type="base",
        agent1_custom_params={
            "initial_trust": 0.8,  # Very trusting
            "memory_length": 20,  # Short memory (forgiving)
            "lr_actor": 2e-4,  # Slower learning
            "noise_scale": 0.05,  # Less exploration
        },
        agent2_custom_params={
            "initial_trust": -0.7,  # Very suspicious
            "memory_length": 100,  # Long memory (holds grudges)
            "lr_actor": 8e-4,  # Faster learning
            "noise_scale": 0.2,  # More exploration
        },
        episodes=800,
        steps_per_episode=40,
        payoff_alpha=3.0,  # Custom payoff parameters
        payoff_beta=8.0,
        save_plots=True,
        run_name="example_2_trust_vs_suspicion",
    )

    results = control.run_simulation(config)

    # Analyze trust dynamics
    print(f"\nCustom Agent Results:")
    print(
        f"Trusting agent performance: {results['final_evaluation']['agent1_avg_reward']:.3f}"
    )
    print(
        f"Suspicious agent performance: {results['final_evaluation']['agent2_avg_reward']:.3f}"
    )

    return results


def example_3_comparison_study():
    """Example 3: Systematic comparison of all agent types."""
    print("=" * 60)
    print("EXAMPLE 3: Systematic Agent Type Comparison")
    print("=" * 60)

    control = SACSControlCenter()

    # Configuration for comparison study
    base_config = SimulationConfig(
        agent1_type="base",  # Will be overridden
        agent2_type="base",  # Will be overridden
        episodes=300,
        steps_per_episode=25,
        save_models=True,  # Don't save models for comparison
        save_plots=True,  # Generate summary plots instead
        output_dir="./comparison_results",
    )

    # Compare all agent types
    agent_types = ["cooperative", "competitive", "adaptive", "cautious"]

    comparison_results = control.run_comparison_study(
        agent_types=agent_types,
        base_config=base_config,
        num_runs=3,  # Run each pairing 3 times
    )

    # Analyze results
    print(f"\nComparison Study Results:")
    for pair, runs in comparison_results.items():
        avg_rewards = []
        for run in runs:
            eval_result = run["final_evaluation"]
            total_reward = (
                eval_result["agent1_avg_reward"] + eval_result["agent2_avg_reward"]
            )
            avg_rewards.append(total_reward)

        mean_performance = sum(avg_rewards) / len(avg_rewards)
        print(f"{pair}: Average total reward = {mean_performance:.3f}")

    return comparison_results


def example_4_parameter_sweep():
    """Example 4: Parameter sweep to understand payoff function effects."""
    print("=" * 60)
    print("EXAMPLE 4: Payoff Parameter Sweep")
    print("=" * 60)

    control = SACSControlCenter()

    # Test different payoff parameters
    alpha_values = [2.0, 4.0, 6.0]  # Mismatch penalty
    beta_values = [5.0, 10.0, 15.0]  # Risk penalty

    sweep_results = {}

    for alpha in alpha_values:
        for beta in beta_values:
            print(f"Testing α={alpha}, β={beta}")

            config = SimulationConfig(
                agent1_type="adaptive",
                agent2_type="adaptive",
                episodes=400,
                payoff_alpha=alpha,
                payoff_beta=beta,
                save_models=True,
                save_plots=True,
                run_name=f"sweep_alpha_{alpha}_beta_{beta}",
            )

            result = control.run_simulation(config)

            # Store key metrics
            final_eval = result["final_evaluation"]
            sweep_results[f"α={alpha}_β={beta}"] = {
                "total_reward": final_eval["agent1_avg_reward"]
                + final_eval["agent2_avg_reward"],
                "agent1_reward": final_eval["agent1_avg_reward"],
                "agent2_reward": final_eval["agent2_avg_reward"],
                "alpha": alpha,
                "beta": beta,
            }

    # Print parameter sweep results
    print(f"\nParameter Sweep Results:")
    print("Configuration\t\tTotal Reward\tAgent1\tAgent2")
    print("-" * 60)
    for config_name, metrics in sweep_results.items():
        print(
            f"{config_name}\t{metrics['total_reward']:.3f}\t\t{metrics['agent1_reward']:.3f}\t{metrics['agent2_reward']:.3f}"
        )

    return sweep_results


def example_5_longitudinal_analysis():
    """Example 5: Longitudinal analysis of agent development."""
    print("=" * 60)
    print("EXAMPLE 5: Longitudinal Development Analysis")
    print("=" * 60)

    control = SACSControlCenter()

    # Run a longer simulation to see development patterns
    config = SimulationConfig(
        agent1_type="adaptive",
        agent2_type="cautious",
        episodes=2000,  # Longer simulation
        steps_per_episode=50,
        evaluation_frequency=200,  # More frequent evaluation
        save_frequency=500,
        save_models=True,
        save_plots=True,
        run_name="example_5_longitudinal_development",
    )

    results = control.run_simulation(config)

    # Analyze learning phases
    episode_rewards = results["training_results"]["episode_rewards"]
    agent1_rewards = episode_rewards["agent1"]
    agent2_rewards = episode_rewards["agent2"]

    # Divide into phases
    phase_size = len(agent1_rewards) // 4
    phases = ["Early", "Mid-Early", "Mid-Late", "Late"]

    print(f"\nLongitudinal Analysis:")
    print("Phase\t\tAgent1 Avg\tAgent2 Avg\tDifference")
    print("-" * 60)

    for i, phase in enumerate(phases):
        start_idx = i * phase_size
        end_idx = (i + 1) * phase_size if i < 3 else len(agent1_rewards)

        phase_rewards1 = agent1_rewards[start_idx:end_idx]
        phase_rewards2 = agent2_rewards[start_idx:end_idx]

        avg1 = sum(phase_rewards1) / len(phase_rewards1)
        avg2 = sum(phase_rewards2) / len(phase_rewards2)
        diff = avg1 - avg2

        print(f"{phase}\t\t{avg1:.3f}\t\t{avg2:.3f}\t\t{diff:+.3f}")

    return results


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import os
from datetime import datetime
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu, friedmanchisquare
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import warnings

warnings.filterwarnings("ignore")


def example_6_interactive_demo():
    """Example 6: Parameter sweep to understand payoff function effects across various different pairings."""
    print("=" * 60)
    print("EXAMPLE 6: Payoff Parameter & Personality Sweep")
    print("=" * 60)

    control = SACSControlCenter()

    # Test different payoff parameters
    alpha_values = [2.0, 4.0, 6.0]  # Mismatch penalty
    beta_values = [0]  # Risk penalty
    personalities = ["cooperative", "competitive", "adaptive", "cautious"]

    sweep_results = {}
    total_combinations = len(personalities) ** 2 * len(alpha_values) * len(beta_values)
    current_combination = 0

    print(f"Running {total_combinations} simulations...")

    for agent1_type in personalities:
        for agent2_type in personalities:
            for alpha in alpha_values:
                for beta in beta_values:
                    current_combination += 1
                    print(
                        f"[{current_combination}/{total_combinations}] Testing a1={agent1_type}, a2={agent2_type}, α={alpha}, β={beta}"
                    )

                    config = SimulationConfig(
                        agent1_type=agent1_type,
                        agent2_type=agent2_type,
                        episodes=200,
                        payoff_alpha=alpha,
                        payoff_beta=beta,
                        save_models=True,
                        save_plots=True,
                        run_name=f"sweep_{agent1_type}_{agent2_type}_alpha_{alpha}_beta_{beta}",
                    )

                    result = control.run_simulation(config)
                    final_eval = result["final_evaluation"]

                    # Use a unique key that captures all variables
                    key = f"{agent1_type}-{agent2_type}_α={alpha}_β={beta}"
                    sweep_results[key] = {
                        "total_reward": final_eval["agent1_avg_reward"]
                        + final_eval["agent2_avg_reward"],
                        "agent1_reward": final_eval["agent1_avg_reward"],
                        "agent2_reward": final_eval["agent2_avg_reward"],
                        "reward_difference": abs(
                            final_eval["agent1_avg_reward"]
                            - final_eval["agent2_avg_reward"]
                        ),
                        "agent1_type": agent1_type,
                        "agent2_type": agent2_type,
                        "personality_pairing": f"{agent1_type}-{agent2_type}",
                        "alpha": alpha,
                        "beta": beta,
                    }

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame.from_dict(sweep_results, orient="index")

    # Print basic results
    print_basic_results(df)

    # Generate comprehensive analysis and visualizations
    analysis_results = analyze_parameter_sweep(df)

    # Perform rigorous statistical testing
    statistical_results = perform_statistical_analysis(df)

    # Save results to CSV
    save_results_to_csv(df, sweep_results, statistical_results)

    return sweep_results, df, analysis_results, statistical_results


def print_basic_results(df):
    """Print basic tabular results"""
    print(f"\nParameter Sweep Results:")
    print("Personality Pair\t\tParameters\t\tTotal\tAgent1\tAgent2\tDiff")
    print("-" * 85)

    for _, row in df.iterrows():
        personality_pair = f"{row['agent1_type']}-{row['agent2_type']}"
        params = f"α={row['alpha']}, β={row['beta']}"
        print(
            f"{personality_pair:<20}\t{params:<15}\t{row['total_reward']:.3f}\t{row['agent1_reward']:.3f}\t{row['agent2_reward']:.3f}\t{row['reward_difference']:.3f}"
        )


def analyze_parameter_sweep(df):
    """Comprehensive analysis of parameter sweep results"""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE ANALYSIS")
    print("=" * 60)

    analysis_results = {}

    # 1. Find optimal parameter combinations - SEPARATE ANALYSIS
    analysis_results["optimal_total_reward"] = find_optimal_total_reward(df)
    analysis_results["optimal_balance"] = find_optimal_balance(df)

    # 2. Analyze personality adaptiveness
    analysis_results["personality_adaptiveness"] = analyze_personality_adaptiveness(df)

    # 3. Generate separate visualizations
    analysis_results["plots"] = create_separate_visualizations(df)

    return analysis_results


def find_optimal_total_reward(df):
    """Find optimal parameter combinations for TOTAL REWARD only"""
    print("\n🏆 OPTIMAL PARAMETERS FOR TOTAL REWARD:")
    print("-" * 50)

    optimal_params = {}

    # Best single combination
    best_total = df.loc[df["total_reward"].idxmax()]
    optimal_params["best_single"] = {
        "alpha": best_total["alpha"],
        "beta": best_total["beta"],
        "pairing": best_total["personality_pairing"],
        "total_reward": best_total["total_reward"],
    }
    print(f"🥇 Best Single Result: α={best_total['alpha']}, β={best_total['beta']}")
    print(
        f"   Pairing: {best_total['personality_pairing']}, Total: {best_total['total_reward']:.3f}"
    )

    # Best average performance across all personality pairings
    param_performance = (
        df.groupby(["alpha", "beta"])
        .agg({"total_reward": ["mean", "std", "min", "max", "count"]})
        .round(3)
    )
    param_performance.columns = ["mean", "std", "min", "max", "count"]

    best_avg_idx = param_performance["mean"].idxmax()
    best_avg = param_performance.loc[best_avg_idx]
    optimal_params["best_average"] = {
        "alpha": best_avg_idx[0],
        "beta": best_avg_idx[1],
        "avg_total_reward": best_avg["mean"],
        "std": best_avg["std"],
        "min": best_avg["min"],
        "max": best_avg["max"],
    }
    print(
        f"📈 Best Average Across All Pairings: α={best_avg_idx[0]}, β={best_avg_idx[1]}"
    )
    print(f"   Mean: {best_avg['mean']:.3f} ± {best_avg['std']:.3f}")
    print(f"   Range: {best_avg['min']:.3f} - {best_avg['max']:.3f}")

    # Show top 5 parameter combinations
    print(f"\n🔝 Top 5 Parameter Combinations (by average total reward):")
    top_5 = param_performance.sort_values("mean", ascending=False).head(5)
    for i, ((alpha, beta), row) in enumerate(top_5.iterrows(), 1):
        print(f"   {i}. α={alpha}, β={beta}: {row['mean']:.3f} ± {row['std']:.3f}")

    return optimal_params


def find_optimal_balance(df):
    """Find optimal parameter combinations for BALANCE (fairness) only"""
    print("\n⚖️ OPTIMAL PARAMETERS FOR BALANCE (FAIRNESS):")
    print("-" * 50)

    balance_params = {}

    # Most balanced single result
    most_balanced = df.loc[df["reward_difference"].idxmin()]
    balance_params["most_balanced_single"] = {
        "alpha": most_balanced["alpha"],
        "beta": most_balanced["beta"],
        "pairing": most_balanced["personality_pairing"],
        "difference": most_balanced["reward_difference"],
        "total_reward": most_balanced["total_reward"],
    }
    print(
        f"🥇 Most Balanced Single Result: α={most_balanced['alpha']}, β={most_balanced['beta']}"
    )
    print(
        f"   Pairing: {most_balanced['personality_pairing']}, Diff: {most_balanced['reward_difference']:.3f}"
    )
    print(f"   (Total reward: {most_balanced['total_reward']:.3f})")

    # Best average balance across all personality pairings
    balance_performance = (
        df.groupby(["alpha", "beta"])
        .agg({"reward_difference": ["mean", "std", "min", "max", "count"]})
        .round(3)
    )
    balance_performance.columns = ["mean", "std", "min", "max", "count"]

    best_balance_idx = balance_performance[
        "mean"
    ].idxmin()  # Lower difference = more balanced
    best_balance = balance_performance.loc[best_balance_idx]
    balance_params["best_average_balance"] = {
        "alpha": best_balance_idx[0],
        "beta": best_balance_idx[1],
        "avg_difference": best_balance["mean"],
        "std": best_balance["std"],
        "min": best_balance["min"],
        "max": best_balance["max"],
    }
    print(f"📈 Best Average Balance: α={best_balance_idx[0]}, β={best_balance_idx[1]}")
    print(f"   Mean Difference: {best_balance['mean']:.3f} ± {best_balance['std']:.3f}")
    print(f"   Range: {best_balance['min']:.3f} - {best_balance['max']:.3f}")

    # Show top 5 most balanced parameter combinations
    print(f"\n🔝 Top 5 Most Balanced Parameter Combinations:")
    top_5_balanced = balance_performance.sort_values("mean", ascending=True).head(5)
    for i, ((alpha, beta), row) in enumerate(top_5_balanced.iterrows(), 1):
        print(f"   {i}. α={alpha}, β={beta}: {row['mean']:.3f} ± {row['std']:.3f}")

    return balance_params


def analyze_personality_adaptiveness(df):
    """Analyze which personality types are most adaptive and provide reasoning"""
    print("\n🎭 PERSONALITY ADAPTIVENESS ANALYSIS:")
    print("-" * 50)

    adaptiveness_results = {}

    # Calculate adaptiveness metrics for each personality type
    personality_metrics = {}

    for personality in df["agent1_type"].unique():
        # Get all instances where this personality appears (as agent1 or agent2)
        as_agent1 = df[df["agent1_type"] == personality]["agent1_reward"]
        as_agent2 = df[df["agent2_type"] == personality]["agent2_reward"]
        all_rewards = pd.concat([as_agent1, as_agent2])

        # Calculate adaptiveness metrics
        personality_metrics[personality] = {
            "mean_reward": all_rewards.mean(),
            "std_reward": all_rewards.std(),
            "min_reward": all_rewards.min(),
            "max_reward": all_rewards.max(),
            "coefficient_of_variation": all_rewards.std()
            / all_rewards.mean(),  # Lower = more consistent
            "range": all_rewards.max() - all_rewards.min(),
            "sample_size": len(all_rewards),
        }

    # Rank personalities by different adaptiveness criteria
    adaptiveness_rankings = {}

    # 1. Highest average performance
    avg_ranking = sorted(
        personality_metrics.items(), key=lambda x: x[1]["mean_reward"], reverse=True
    )
    adaptiveness_rankings["highest_average"] = avg_ranking

    # 2. Most consistent (lowest coefficient of variation)
    consistency_ranking = sorted(
        personality_metrics.items(), key=lambda x: x[1]["coefficient_of_variation"]
    )
    adaptiveness_rankings["most_consistent"] = consistency_ranking

    # 3. Highest minimum performance (robust floor)
    floor_ranking = sorted(
        personality_metrics.items(), key=lambda x: x[1]["min_reward"], reverse=True
    )
    adaptiveness_rankings["highest_floor"] = floor_ranking

    # 4. Composite adaptiveness score
    # Normalize metrics and create composite score
    for personality in personality_metrics:
        metrics = personality_metrics[personality]
        # Higher mean reward, lower CV, higher min reward = more adaptive
        normalized_mean = (
            metrics["mean_reward"]
            - min([p["mean_reward"] for p in personality_metrics.values()])
        ) / (
            max([p["mean_reward"] for p in personality_metrics.values()])
            - min([p["mean_reward"] for p in personality_metrics.values()])
        )
        normalized_consistency = 1 - (
            (
                metrics["coefficient_of_variation"]
                - min(
                    [
                        p["coefficient_of_variation"]
                        for p in personality_metrics.values()
                    ]
                )
            )
            / (
                max(
                    [
                        p["coefficient_of_variation"]
                        for p in personality_metrics.values()
                    ]
                )
                - min(
                    [
                        p["coefficient_of_variation"]
                        for p in personality_metrics.values()
                    ]
                )
            )
        )
        normalized_floor = (
            metrics["min_reward"]
            - min([p["min_reward"] for p in personality_metrics.values()])
        ) / (
            max([p["min_reward"] for p in personality_metrics.values()])
            - min([p["min_reward"] for p in personality_metrics.values()])
        )

        # Composite score (equal weights)
        metrics["composite_adaptiveness"] = (
            normalized_mean + normalized_consistency + normalized_floor
        ) / 3

    composite_ranking = sorted(
        personality_metrics.items(),
        key=lambda x: x[1]["composite_adaptiveness"],
        reverse=True,
    )
    adaptiveness_rankings["composite"] = composite_ranking

    # Display results
    print("📊 Personality Performance Metrics:")
    print("Personality\t\tMean\tStd\tCV\tMin\tMax\tComposite")
    print("-" * 70)
    for personality, metrics in personality_metrics.items():
        print(
            f"{personality:<15}\t{metrics['mean_reward']:.3f}\t{metrics['std_reward']:.3f}\t{metrics['coefficient_of_variation']:.3f}\t{metrics['min_reward']:.3f}\t{metrics['max_reward']:.3f}\t{metrics['composite_adaptiveness']:.3f}"
        )

    print("\n🏆 ADAPTIVENESS RANKINGS & REASONING:")
    print("-" * 50)

    print("1️⃣ HIGHEST AVERAGE PERFORMANCE:")
    for i, (personality, metrics) in enumerate(avg_ranking, 1):
        print(f"   {i}. {personality}: {metrics['mean_reward']:.3f}")

    print("\n2️⃣ MOST CONSISTENT (Lowest Variability):")
    for i, (personality, metrics) in enumerate(consistency_ranking, 1):
        print(f"   {i}. {personality}: CV = {metrics['coefficient_of_variation']:.3f}")

    print("\n3️⃣ HIGHEST PERFORMANCE FLOOR (Robustness):")
    for i, (personality, metrics) in enumerate(floor_ranking, 1):
        print(f"   {i}. {personality}: Min = {metrics['min_reward']:.3f}")

    print("\n🎯 COMPOSITE ADAPTIVENESS RANKING:")
    for i, (personality, metrics) in enumerate(composite_ranking, 1):
        print(f"   {i}. {personality}: Score = {metrics['composite_adaptiveness']:.3f}")

    # Provide reasoning for most adaptive personality
    most_adaptive = composite_ranking[0][0]
    most_adaptive_metrics = composite_ranking[0][1]

    print(f"\n💡 SUGGESTED MOST ADAPTIVE PERSONALITY: {most_adaptive.upper()}")
    print("REASONING:")
    print(
        f"   • Average Performance: {most_adaptive_metrics['mean_reward']:.3f} (rank: {[p[0] for p in avg_ranking].index(most_adaptive) + 1})"
    )
    print(
        f"   • Consistency (CV): {most_adaptive_metrics['coefficient_of_variation']:.3f} (rank: {[p[0] for p in consistency_ranking].index(most_adaptive) + 1})"
    )
    print(
        f"   • Performance Floor: {most_adaptive_metrics['min_reward']:.3f} (rank: {[p[0] for p in floor_ranking].index(most_adaptive) + 1})"
    )
    print(
        f"   • Composite Score: {most_adaptive_metrics['composite_adaptiveness']:.3f}"
    )

    adaptiveness_results["metrics"] = personality_metrics
    adaptiveness_results["rankings"] = adaptiveness_rankings
    adaptiveness_results["most_adaptive"] = most_adaptive

    return adaptiveness_results


def perform_statistical_analysis(df):
    """Perform rigorous statistical testing with multiple comparison corrections"""
    print("\n" + "=" * 60)
    print("RIGOROUS STATISTICAL ANALYSIS")
    print("=" * 60)

    statistical_results = {}
    all_p_values = []
    test_descriptions = []

    print(
        "🔬 Performing comprehensive statistical testing with multiple comparison corrections..."
    )

    # 1. Test for main effects of alpha parameter
    print("\n1️⃣ ALPHA PARAMETER MAIN EFFECT:")
    alpha_groups = [
        df[df["alpha"] == alpha]["total_reward"].values
        for alpha in df["alpha"].unique()
    ]
    alpha_stat, alpha_p = kruskal(*alpha_groups)
    all_p_values.append(alpha_p)
    test_descriptions.append("Alpha parameter main effect (Kruskal-Wallis)")
    print(f"   Kruskal-Wallis H = {alpha_stat:.4f}, p = {alpha_p:.6f}")

    # 2. Test for main effects of beta parameter
    print("\n2️⃣ BETA PARAMETER MAIN EFFECT:")
    beta_groups = [
        df[df["beta"] == beta]["total_reward"].values for beta in df["beta"].unique()
    ]
    beta_stat, beta_p = kruskal(*beta_groups)
    all_p_values.append(beta_p)
    test_descriptions.append("Beta parameter main effect (Kruskal-Wallis)")
    print(f"   Kruskal-Wallis H = {beta_stat:.4f}, p = {beta_p:.6f}")

    # 3. Test for personality pairing effects
    print("\n3️⃣ PERSONALITY PAIRING EFFECTS:")
    pairing_groups = [
        df[df["personality_pairing"] == pairing]["total_reward"].values
        for pairing in df["personality_pairing"].unique()
    ]
    pairing_stat, pairing_p = kruskal(*pairing_groups)
    all_p_values.append(pairing_p)
    test_descriptions.append("Personality pairing effect (Kruskal-Wallis)")
    print(f"   Kruskal-Wallis H = {pairing_stat:.4f}, p = {pairing_p:.6f}")

    # 4. Pairwise comparisons for alpha values (if main effect significant)
    print("\n4️⃣ PAIRWISE ALPHA COMPARISONS:")
    alpha_values = sorted(df["alpha"].unique())
    alpha_pairwise_p = []
    for i in range(len(alpha_values)):
        for j in range(i + 1, len(alpha_values)):
            group1 = df[df["alpha"] == alpha_values[i]]["total_reward"].values
            group2 = df[df["alpha"] == alpha_values[j]]["total_reward"].values
            stat, p = mannwhitneyu(group1, group2, alternative="two-sided")
            alpha_pairwise_p.append(p)
            all_p_values.append(p)
            test_descriptions.append(
                f"Alpha {alpha_values[i]} vs {alpha_values[j]} (Mann-Whitney U)"
            )
            print(
                f"   α={alpha_values[i]} vs α={alpha_values[j]}: U = {stat:.4f}, p = {p:.6f}"
            )

    # 5. Pairwise comparisons for beta values (if main effect significant)
    print("\n5️⃣ PAIRWISE BETA COMPARISONS:")
    beta_values = sorted(df["beta"].unique())
    beta_pairwise_p = []
    for i in range(len(beta_values)):
        for j in range(i + 1, len(beta_values)):
            group1 = df[df["beta"] == beta_values[i]]["total_reward"].values
            group2 = df[df["beta"] == beta_values[j]]["total_reward"].values
            stat, p = mannwhitneyu(group1, group2, alternative="two-sided")
            beta_pairwise_p.append(p)
            all_p_values.append(p)
            test_descriptions.append(
                f"Beta {beta_values[i]} vs {beta_values[j]} (Mann-Whitney U)"
            )
            print(
                f"   β={beta_values[i]} vs β={beta_values[j]}: U = {stat:.4f}, p = {p:.6f}"
            )

    # 6. Test for parameter interaction effects (2-way ANOVA)
    print("\n6️⃣ PARAMETER INTERACTION ANALYSIS (2-way ANOVA):")
    try:
        # Convert to categorical for ANOVA
        df_anova = df.copy()
        df_anova["alpha_cat"] = df_anova["alpha"].astype(str)
        df_anova["beta_cat"] = df_anova["beta"].astype(str)

        # Fit ANOVA model
        model = ols(
            "total_reward ~ C(alpha_cat) + C(beta_cat) + C(alpha_cat):C(beta_cat)",
            data=df_anova,
        ).fit()
        anova_results = anova_lm(model, typ=2)

        print("   ANOVA Results:")
        print(anova_results)

        # Extract p-values from ANOVA
        anova_p_values = anova_results["PR(>F)"].dropna().values
        all_p_values.extend(anova_p_values)
        test_descriptions.extend(
            [
                "Alpha main effect (ANOVA)",
                "Beta main effect (ANOVA)",
                "Alpha × Beta interaction (ANOVA)",
            ]
        )

    except Exception as e:
        print(f"   ANOVA analysis failed: {e}")

    # 7. Test for balance (fairness) effects
    print("\n7️⃣ BALANCE (FAIRNESS) ANALYSIS:")

    # Alpha effect on balance
    alpha_balance_groups = [
        df[df["alpha"] == alpha]["reward_difference"].values
        for alpha in df["alpha"].unique()
    ]
    alpha_balance_stat, alpha_balance_p = kruskal(*alpha_balance_groups)
    all_p_values.append(alpha_balance_p)
    test_descriptions.append("Alpha effect on balance (Kruskal-Wallis)")
    print(
        f"   Alpha effect on balance: H = {alpha_balance_stat:.4f}, p = {alpha_balance_p:.6f}"
    )

    # Beta effect on balance
    beta_balance_groups = [
        df[df["beta"] == beta]["reward_difference"].values
        for beta in df["beta"].unique()
    ]
    beta_balance_stat, beta_balance_p = kruskal(*beta_balance_groups)
    all_p_values.append(beta_balance_p)
    test_descriptions.append("Beta effect on balance (Kruskal-Wallis)")
    print(
        f"   Beta effect on balance: H = {beta_balance_stat:.4f}, p = {beta_balance_p:.6f}"
    )

    # Personality pairing effect on balance
    pairing_balance_groups = [
        df[df["personality_pairing"] == pairing]["reward_difference"].values
        for pairing in df["personality_pairing"].unique()
    ]
    pairing_balance_stat, pairing_balance_p = kruskal(*pairing_balance_groups)
    all_p_values.append(pairing_balance_p)
    test_descriptions.append("Personality pairing effect on balance (Kruskal-Wallis)")
    print(
        f"   Personality pairing effect on balance: H = {pairing_balance_stat:.4f}, p = {pairing_balance_p:.6f}"
    )

    # 8. MULTIPLE COMPARISON CORRECTION
    print("\n8️⃣ MULTIPLE COMPARISON CORRECTION:")
    print(f"   Total number of statistical tests performed: {len(all_p_values)}")

    # Apply Benjamini-Hochberg FDR correction
    rejected_bh, corrected_p_bh, alpha_sidak, alpha_bonf = multipletests(
        all_p_values, alpha=0.05, method="fdr_bh", is_sorted=False, returnsorted=False
    )

    # Apply Bonferroni correction
    rejected_bonf, corrected_p_bonf, _, _ = multipletests(
        all_p_values,
        alpha=0.05,
        method="bonferroni",
        is_sorted=False,
        returnsorted=False,
    )

    print(f"   Benjamini-Hochberg FDR α = {alpha_sidak:.6f}")
    print(f"   Bonferroni corrected α = {alpha_bonf:.6f}")

    # 9. SUMMARY OF SIGNIFICANT RESULTS
    print("\n9️⃣ SIGNIFICANT RESULTS SUMMARY:")
    print("   After Benjamini-Hochberg FDR correction:")
    significant_bh = sum(rejected_bh)
    print(f"   {significant_bh}/{len(all_p_values)} tests remain significant")

    if significant_bh > 0:
        print("   Significant tests (FDR corrected):")
        for i, (rejected, p_orig, p_corr, desc) in enumerate(
            zip(rejected_bh, all_p_values, corrected_p_bh, test_descriptions)
        ):
            if rejected:
                print(
                    f"      • {desc}: p_original = {p_orig:.6f}, p_corrected = {p_corr:.6f}"
                )

    print("   After Bonferroni correction:")
    significant_bonf = sum(rejected_bonf)
    print(f"   {significant_bonf}/{len(all_p_values)} tests remain significant")

    if significant_bonf > 0:
        print("   Significant tests (Bonferroni corrected):")
        for i, (rejected, p_orig, p_corr, desc) in enumerate(
            zip(rejected_bonf, all_p_values, corrected_p_bonf, test_descriptions)
        ):
            if rejected:
                print(
                    f"      • {desc}: p_original = {p_orig:.6f}, p_corrected = {p_corr:.6f}"
                )

    # 10. EFFECT SIZES
    print("\n🔟 EFFECT SIZE ANALYSIS:")

    # Effect size for alpha parameter (eta-squared approximation using Kruskal-Wallis)
    alpha_eta_squared = (alpha_stat - len(df["alpha"].unique()) + 1) / (
        len(df) - len(df["alpha"].unique())
    )
    print(f"   Alpha parameter effect size (η² approximation): {alpha_eta_squared:.4f}")

    # Effect size for beta parameter
    beta_eta_squared = (beta_stat - len(df["beta"].unique()) + 1) / (
        len(df) - len(df["beta"].unique())
    )
    print(f"   Beta parameter effect size (η² approximation): {beta_eta_squared:.4f}")

    # Effect size for personality pairing
    pairing_eta_squared = (
        pairing_stat - len(df["personality_pairing"].unique()) + 1
    ) / (len(df) - len(df["personality_pairing"].unique()))
    print(
        f"   Personality pairing effect size (η² approximation): {pairing_eta_squared:.4f}"
    )

    # Store all results
    statistical_results = {
        "alpha_main_effect": {
            "statistic": alpha_stat,
            "p_value": alpha_p,
            "effect_size": alpha_eta_squared,
        },
        "beta_main_effect": {
            "statistic": beta_stat,
            "p_value": beta_p,
            "effect_size": beta_eta_squared,
        },
        "personality_main_effect": {
            "statistic": pairing_stat,
            "p_value": pairing_p,
            "effect_size": pairing_eta_squared,
        },
        "all_p_values": all_p_values,
        "test_descriptions": test_descriptions,
        "corrected_p_bh": corrected_p_bh,
        "corrected_p_bonf": corrected_p_bonf,
        "rejected_bh": rejected_bh,
        "rejected_bonf": rejected_bonf,
        "significant_tests_bh": significant_bh,
        "significant_tests_bonf": significant_bonf,
        "alpha_balance_effect": {
            "statistic": alpha_balance_stat,
            "p_value": alpha_balance_p,
        },
        "beta_balance_effect": {
            "statistic": beta_balance_stat,
            "p_value": beta_balance_p,
        },
        "pairing_balance_effect": {
            "statistic": pairing_balance_stat,
            "p_value": pairing_balance_p,
        },
    }

    # 11. PRACTICAL SIGNIFICANCE
    print("\n1️⃣1️⃣ PRACTICAL SIGNIFICANCE ASSESSMENT:")

    # Calculate Cohen's d for largest effects
    alpha_groups_means = [
        df[df["alpha"] == alpha]["total_reward"].mean()
        for alpha in df["alpha"].unique()
    ]
    alpha_range = max(alpha_groups_means) - min(alpha_groups_means)
    pooled_std = df["total_reward"].std()
    alpha_cohens_d = alpha_range / pooled_std
    print(f"   Alpha parameter practical effect (Cohen's d): {alpha_cohens_d:.4f}")

    beta_groups_means = [
        df[df["beta"] == beta]["total_reward"].mean() for beta in df["beta"].unique()
    ]
    beta_range = max(beta_groups_means) - min(beta_groups_means)
    beta_cohens_d = beta_range / pooled_std
    print(f"   Beta parameter practical effect (Cohen's d): {beta_cohens_d:.4f}")

    # Interpretation guidelines
    print("\n   Effect Size Interpretation (Cohen's guidelines):")
    print("   • Small effect: d ≈ 0.2, η² ≈ 0.01")
    print("   • Medium effect: d ≈ 0.5, η² ≈ 0.06")
    print("   • Large effect: d ≈ 0.8, η² ≈ 0.14")

    statistical_results["alpha_cohens_d"] = alpha_cohens_d
    statistical_results["beta_cohens_d"] = beta_cohens_d

    return statistical_results


def create_separate_visualizations(df):
    """Create separate, focused visualizations for total reward and balance analysis"""
    print("\n📈 GENERATING SEPARATE VISUALIZATIONS...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_files = {}

    # Set up the plotting style
    plt.style.use("seaborn-v0_8")

    # 1. TOTAL REWARD ANALYSIS PLOTS
    print("   Creating total reward analysis plots...")

    # Total reward heatmap by parameters
    fig, ax = plt.subplots(figsize=(10, 8))
    pivot_total = df.groupby(["alpha", "beta"])["total_reward"].mean().unstack()
    sns.heatmap(
        pivot_total,
        annot=True,
        cmap="viridis",
        fmt=".3f",
        ax=ax,
        cbar_kws={"label": "Average Total Reward"},
    )
    ax.set_title(
        "Total Reward by Parameter Combinations\n(Higher = Better)",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("Beta (Risk Penalty)", fontsize=12)
    ax.set_ylabel("Alpha (Mismatch Penalty)", fontsize=12)
    plt.tight_layout()
    total_reward_heatmap = f"total_reward_heatmap_{timestamp}.png"
    plt.savefig(total_reward_heatmap, dpi=300, bbox_inches="tight")
    plot_files["total_reward_heatmap"] = total_reward_heatmap
    plt.show()

    # Total reward by personality pairing
    fig, ax = plt.subplots(figsize=(12, 8))
    pairing_means = (
        df.groupby("personality_pairing")["total_reward"]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=True)
    )
    bars = ax.barh(
        range(len(pairing_means)),
        pairing_means["mean"],
        xerr=pairing_means["std"],
        capsize=5,
        color="skyblue",
        alpha=0.8,
    )
    ax.set_yticks(range(len(pairing_means)))
    ax.set_yticklabels(pairing_means.index)
    ax.set_xlabel("Average Total Reward", fontsize=12)
    ax.set_title(
        "Total Reward by Personality Pairing\n(with Standard Deviation)",
        fontsize=16,
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)

    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(
            width + pairing_means["std"].iloc[i] + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            ha="left",
            va="center",
            fontweight="bold",
        )

    plt.tight_layout()
    total_reward_pairing = f"total_reward_by_pairing_{timestamp}.png"
    plt.savefig(total_reward_pairing, dpi=300, bbox_inches="tight")
    plot_files["total_reward_pairing"] = total_reward_pairing
    plt.show()

    # Parameter sensitivity plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Alpha sensitivity
    alpha_effects = df.groupby("alpha").agg({"total_reward": ["mean", "std", "count"]})
    alpha_effects.columns = ["mean", "std", "count"]
    ax1.errorbar(
        alpha_effects.index,
        alpha_effects["mean"],
        yerr=alpha_effects["std"],
        marker="o",
        markersize=10,
        linewidth=3,
        capsize=8,
        capthick=2,
    )
    ax1.set_xlabel("Alpha (Mismatch Penalty)", fontsize=12)
    ax1.set_ylabel("Average Total Reward", fontsize=12)
    ax1.set_title(
        "Alpha Parameter Sensitivity\n(with Standard Error)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)

    # Beta sensitivity
    beta_effects = df.groupby("beta").agg({"total_reward": ["mean", "std", "count"]})
    beta_effects.columns = ["mean", "std", "count"]
    ax2.errorbar(
        beta_effects.index,
        beta_effects["mean"],
        yerr=beta_effects["std"],
        marker="s",
        markersize=10,
        linewidth=3,
        capsize=8,
        capthick=2,
        color="orange",
    )
    ax2.set_xlabel("Beta (Risk Penalty)", fontsize=12)
    ax2.set_ylabel("Average Total Reward", fontsize=12)
    ax2.set_title(
        "Beta Parameter Sensitivity\n(with Standard Error)",
        fontsize=14,
        fontweight="bold",
    )
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    parameter_sensitivity = f"parameter_sensitivity_{timestamp}.png"
    plt.savefig(parameter_sensitivity, dpi=300, bbox_inches="tight")
    plot_files["parameter_sensitivity"] = parameter_sensitivity
    plt.show()

    # 2. BALANCE ANALYSIS PLOTS
    print("   Creating balance analysis plots...")

    # Balance heatmap by parameters
    fig, ax = plt.subplots(figsize=(10, 8))
    pivot_balance = df.groupby(["alpha", "beta"])["reward_difference"].mean().unstack()
    sns.heatmap(
        pivot_balance,
        annot=True,
        cmap="RdYlBu_r",
        fmt=".3f",
        ax=ax,
        cbar_kws={"label": "Average Reward Difference"},
    )
    ax.set_title(
        "Balance (Fairness) by Parameter Combinations\n(Lower = More Balanced)",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("Beta (Risk Penalty)", fontsize=12)
    ax.set_ylabel("Alpha (Mismatch Penalty)", fontsize=12)
    plt.tight_layout()
    balance_heatmap = f"balance_heatmap_{timestamp}.png"
    plt.savefig(balance_heatmap, dpi=300, bbox_inches="tight")
    plot_files["balance_heatmap"] = balance_heatmap
    plt.show()

    # Balance by personality pairing
    fig, ax = plt.subplots(figsize=(12, 8))
    balance_means = (
        df.groupby("personality_pairing")["reward_difference"]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=True)
    )
    bars = ax.barh(
        range(len(balance_means)),
        balance_means["mean"],
        xerr=balance_means["std"],
        capsize=5,
        color="lightcoral",
        alpha=0.8,
    )
    ax.set_yticks(range(len(balance_means)))
    ax.set_yticklabels(balance_means.index)
    ax.set_xlabel("Average Reward Difference (Lower = More Balanced)", fontsize=12)
    ax.set_title(
        "Balance by Personality Pairing\n(with Standard Deviation)",
        fontsize=16,
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)

    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(
            width + balance_means["std"].iloc[i] + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            ha="left",
            va="center",
            fontweight="bold",
        )

    plt.tight_layout()
    balance_pairing = f"balance_by_pairing_{timestamp}.png"
    plt.savefig(balance_pairing, dpi=300, bbox_inches="tight")
    plot_files["balance_pairing"] = balance_pairing
    plt.show()

    # 3. PERSONALITY ADAPTIVENESS PLOTS
    print("   Creating personality adaptiveness plots...")

    # Individual personality performance across all conditions
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    personalities = df["agent1_type"].unique()

    for i, personality in enumerate(personalities):
        ax = axes[i // 2, i % 2]

        # Get rewards for this personality (as agent1 and agent2)
        as_agent1 = df[df["agent1_type"] == personality]["agent1_reward"]
        as_agent2 = df[df["agent2_type"] == personality]["agent2_reward"]
        all_rewards = pd.concat([as_agent1, as_agent2])

        # Create distribution plot
        ax.hist(
            all_rewards, bins=15, alpha=0.7, color=plt.cm.Set3(i), edgecolor="black"
        )
        ax.axvline(
            all_rewards.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {all_rewards.mean():.3f}",
        )
        ax.axvline(
            all_rewards.median(),
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"Median: {all_rewards.median():.3f}",
        )

        ax.set_title(
            f"{personality.capitalize()} Personality\nReward Distribution",
            fontweight="bold",
        )
        ax.set_xlabel("Individual Reward")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    personality_distributions = f"personality_distributions_{timestamp}.png"
    plt.savefig(personality_distributions, dpi=300, bbox_inches="tight")
    plot_files["personality_distributions"] = personality_distributions
    plt.show()

    # Personality adaptiveness metrics comparison
    personality_metrics = {}
    for personality in personalities:
        as_agent1 = df[df["agent1_type"] == personality]["agent1_reward"]
        as_agent2 = df[df["agent2_type"] == personality]["agent2_reward"]
        all_rewards = pd.concat([as_agent1, as_agent2])

        personality_metrics[personality] = {
            "mean": all_rewards.mean(),
            "std": all_rewards.std(),
            "min": all_rewards.min(),
            "max": all_rewards.max(),
            "cv": all_rewards.std() / all_rewards.mean(),
        }

    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Mean performance
    means = [personality_metrics[p]["mean"] for p in personalities]
    bars1 = ax1.bar(
        personalities, means, color=plt.cm.Set3(range(len(personalities))), alpha=0.8
    )
    ax1.set_title("Average Performance by Personality", fontweight="bold")
    ax1.set_ylabel("Average Reward")
    ax1.tick_params(axis="x", rotation=45)
    for bar, mean in zip(bars1, means):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Consistency (lower CV = more consistent)
    cvs = [personality_metrics[p]["cv"] for p in personalities]
    bars2 = ax2.bar(
        personalities, cvs, color=plt.cm.Set3(range(len(personalities))), alpha=0.8
    )
    ax2.set_title(
        "Consistency by Personality\n(Lower = More Consistent)", fontweight="bold"
    )
    ax2.set_ylabel("Coefficient of Variation")
    ax2.tick_params(axis="x", rotation=45)
    for bar, cv in zip(bars2, cvs):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{cv:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Performance floor (robustness)
    mins = [personality_metrics[p]["min"] for p in personalities]
    bars3 = ax3.bar(
        personalities, mins, color=plt.cm.Set3(range(len(personalities))), alpha=0.8
    )
    ax3.set_title(
        "Performance Floor by Personality\n(Higher = More Robust)", fontweight="bold"
    )
    ax3.set_ylabel("Minimum Reward")
    ax3.tick_params(axis="x", rotation=45)
    for bar, min_val in zip(bars3, mins):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{min_val:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Performance ceiling (potential)
    maxs = [personality_metrics[p]["max"] for p in personalities]
    bars4 = ax4.bar(
        personalities, maxs, color=plt.cm.Set3(range(len(personalities))), alpha=0.8
    )
    ax4.set_title(
        "Performance Ceiling by Personality\n(Higher = More Potential)",
        fontweight="bold",
    )
    ax4.set_ylabel("Maximum Reward")
    ax4.tick_params(axis="x", rotation=45)
    for bar, max_val in zip(bars4, maxs):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{max_val:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    personality_metrics_plot = f"personality_metrics_{timestamp}.png"
    plt.savefig(personality_metrics_plot, dpi=300, bbox_inches="tight")
    plot_files["personality_metrics"] = personality_metrics_plot
    plt.show()

    # 4. STATISTICAL SIGNIFICANCE VISUALIZATION
    print("   Creating statistical significance visualization...")

    # Create a summary plot of statistical test results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Alpha parameter box plots
    alpha_data = [
        df[df["alpha"] == alpha]["total_reward"].values
        for alpha in sorted(df["alpha"].unique())
    ]
    ax1.boxplot(
        alpha_data, labels=[f"α={alpha}" for alpha in sorted(df["alpha"].unique())]
    )
    ax1.set_title("Total Reward Distribution by Alpha Parameter", fontweight="bold")
    ax1.set_ylabel("Total Reward")
    ax1.grid(True, alpha=0.3)

    # Beta parameter box plots
    beta_data = [
        df[df["beta"] == beta]["total_reward"].values
        for beta in sorted(df["beta"].unique())
    ]
    ax2.boxplot(beta_data, labels=[f"β={beta}" for beta in sorted(df["beta"].unique())])
    ax2.set_title("Total Reward Distribution by Beta Parameter", fontweight="bold")
    ax2.set_ylabel("Total Reward")
    ax2.grid(True, alpha=0.3)

    # Personality pairing box plots (top 8 most different pairings)
    pairing_variance = (
        df.groupby("personality_pairing")["total_reward"]
        .var()
        .sort_values(ascending=False)
    )
    top_pairings = pairing_variance.head(8).index
    pairing_data = [
        df[df["personality_pairing"] == pairing]["total_reward"].values
        for pairing in top_pairings
    ]
    ax3.boxplot(
        pairing_data,
        labels=[pairing.replace("-", "\nvs\n") for pairing in top_pairings],
    )
    ax3.set_title(
        "Total Reward Distribution\n(Top 8 Most Variable Pairings)", fontweight="bold"
    )
    ax3.set_ylabel("Total Reward")
    ax3.tick_params(axis="x", rotation=45)
    ax3.grid(True, alpha=0.3)

    # Balance distribution by parameters
    balance_alpha_data = [
        df[df["alpha"] == alpha]["reward_difference"].values
        for alpha in sorted(df["alpha"].unique())
    ]
    ax4.boxplot(
        balance_alpha_data,
        labels=[f"α={alpha}" for alpha in sorted(df["alpha"].unique())],
    )
    ax4.set_title(
        "Balance (Fairness) Distribution by Alpha Parameter", fontweight="bold"
    )
    ax4.set_ylabel("Reward Difference")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    statistical_plots = f"statistical_distributions_{timestamp}.png"
    plt.savefig(statistical_plots, dpi=300, bbox_inches="tight")
    plot_files["statistical_plots"] = statistical_plots
    plt.show()

    print(f"   📊 All visualization files saved with timestamp: {timestamp}")

    return plot_files


def save_results_to_csv(df, sweep_results, statistical_results):
    """Save comprehensive results to CSV files including statistical analysis"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Save detailed results
    csv_filename = f"parameter_sweep_results_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"💾 Detailed results saved to: {csv_filename}")

    # 2. Save summary statistics by parameter combinations
    summary_filename = f"parameter_sweep_summary_{timestamp}.csv"
    summary_stats = (
        df.groupby(["alpha", "beta"])
        .agg(
            {
                "total_reward": ["mean", "std", "min", "max", "count"],
                "reward_difference": ["mean", "std", "min", "max"],
                "agent1_reward": ["mean", "std"],
                "agent2_reward": ["mean", "std"],
            }
        )
        .round(4)
    )

    # Flatten column names
    summary_stats.columns = [
        "_".join(col).strip() for col in summary_stats.columns.values
    ]
    summary_stats.to_csv(summary_filename)
    print(f"📋 Summary statistics saved to: {summary_filename}")

    # 3. Save personality-specific analysis
    personality_filename = f"personality_analysis_{timestamp}.csv"
    personality_data = []

    for personality in df["agent1_type"].unique():
        as_agent1 = df[df["agent1_type"] == personality]["agent1_reward"]
        as_agent2 = df[df["agent2_type"] == personality]["agent2_reward"]
        all_rewards = pd.concat([as_agent1, as_agent2])

        personality_data.append(
            {
                "personality": personality,
                "mean_reward": all_rewards.mean(),
                "std_reward": all_rewards.std(),
                "min_reward": all_rewards.min(),
                "max_reward": all_rewards.max(),
                "coefficient_of_variation": all_rewards.std() / all_rewards.mean(),
                "sample_size": len(all_rewards),
                "median_reward": all_rewards.median(),
                "q25_reward": all_rewards.quantile(0.25),
                "q75_reward": all_rewards.quantile(0.75),
            }
        )

    personality_df = pd.DataFrame(personality_data)
    personality_df.to_csv(personality_filename, index=False)
    print(f"🎭 Personality analysis saved to: {personality_filename}")

    # 4. Save statistical test results
    stats_filename = f"statistical_results_{timestamp}.csv"
    stats_data = []

    # Main effects
    stats_data.append(
        {
            "test_name": "Alpha Parameter Main Effect",
            "test_type": "Kruskal-Wallis",
            "statistic": statistical_results["alpha_main_effect"]["statistic"],
            "p_value": statistical_results["alpha_main_effect"]["p_value"],
            "effect_size": statistical_results["alpha_main_effect"]["effect_size"],
            "interpretation": "Effect of alpha parameter on total reward",
        }
    )

    stats_data.append(
        {
            "test_name": "Beta Parameter Main Effect",
            "test_type": "Kruskal-Wallis",
            "statistic": statistical_results["beta_main_effect"]["statistic"],
            "p_value": statistical_results["beta_main_effect"]["p_value"],
            "effect_size": statistical_results["beta_main_effect"]["effect_size"],
            "interpretation": "Effect of beta parameter on total reward",
        }
    )

    stats_data.append(
        {
            "test_name": "Personality Pairing Main Effect",
            "test_type": "Kruskal-Wallis",
            "statistic": statistical_results["personality_main_effect"]["statistic"],
            "p_value": statistical_results["personality_main_effect"]["p_value"],
            "effect_size": statistical_results["personality_main_effect"][
                "effect_size"
            ],
            "interpretation": "Effect of personality pairing on total reward",
        }
    )

    # Balance effects
    stats_data.append(
        {
            "test_name": "Alpha Effect on Balance",
            "test_type": "Kruskal-Wallis",
            "statistic": statistical_results["alpha_balance_effect"]["statistic"],
            "p_value": statistical_results["alpha_balance_effect"]["p_value"],
            "effect_size": "N/A",
            "interpretation": "Effect of alpha parameter on reward balance",
        }
    )

    stats_data.append(
        {
            "test_name": "Beta Effect on Balance",
            "test_type": "Kruskal-Wallis",
            "statistic": statistical_results["beta_balance_effect"]["statistic"],
            "p_value": statistical_results["beta_balance_effect"]["p_value"],
            "effect_size": "N/A",
            "interpretation": "Effect of beta parameter on reward balance",
        }
    )

    stats_data.append(
        {
            "test_name": "Personality Pairing Effect on Balance",
            "test_type": "Kruskal-Wallis",
            "statistic": statistical_results["pairing_balance_effect"]["statistic"],
            "p_value": statistical_results["pairing_balance_effect"]["p_value"],
            "effect_size": "N/A",
            "interpretation": "Effect of personality pairing on reward balance",
        }
    )

    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(stats_filename, index=False)
    print(f"🔬 Statistical results saved to: {stats_filename}")

    # 5. Save multiple comparison corrections
    corrections_filename = f"multiple_comparisons_{timestamp}.csv"
    corrections_data = []

    for i, (desc, p_orig, p_bh, p_bonf, sig_bh, sig_bonf) in enumerate(
        zip(
            statistical_results["test_descriptions"],
            statistical_results["all_p_values"],
            statistical_results["corrected_p_bh"],
            statistical_results["corrected_p_bonf"],
            statistical_results["rejected_bh"],
            statistical_results["rejected_bonf"],
        )
    ):

        corrections_data.append(
            {
                "test_description": desc,
                "original_p_value": p_orig,
                "benjamini_hochberg_p": p_bh,
                "bonferroni_p": p_bonf,
                "significant_bh": sig_bh,
                "significant_bonferroni": sig_bonf,
            }
        )

    corrections_df = pd.DataFrame(corrections_data)
    corrections_df.to_csv(corrections_filename, index=False)
    print(f"📊 Multiple comparison corrections saved to: {corrections_filename}")

    # 6. Save optimal parameter recommendations
    optimal_filename = f"optimal_parameters_{timestamp}.csv"
    optimal_data = []

    # Best for total reward
    param_combinations = (
        df.groupby(["alpha", "beta"])
        .agg({"total_reward": ["mean", "std"], "reward_difference": ["mean", "std"]})
        .round(4)
    )

    for (alpha, beta), group in param_combinations.iterrows():
        optimal_data.append(
            {
                "alpha": alpha,
                "beta": beta,
                "avg_total_reward": group[("total_reward", "mean")],
                "std_total_reward": group[("total_reward", "std")],
                "avg_balance": group[("reward_difference", "mean")],
                "std_balance": group[("reward_difference", "std")],
                "rank_by_total_reward": None,  # Will be filled below
                "rank_by_balance": None,  # Will be filled below
            }
        )

    optimal_df = pd.DataFrame(optimal_data)
    optimal_df["rank_by_total_reward"] = optimal_df["avg_total_reward"].rank(
        ascending=False
    )
    optimal_df["rank_by_balance"] = optimal_df["avg_balance"].rank(
        ascending=True
    )  # Lower difference = better rank
    optimal_df = optimal_df.sort_values("rank_by_total_reward")
    optimal_df.to_csv(optimal_filename, index=False)
    print(f"🎯 Optimal parameters ranked saved to: {optimal_filename}")

    return {
        "detailed_results": csv_filename,
        "summary_stats": summary_filename,
        "personality_analysis": personality_filename,
        "statistical_results": stats_filename,
        "multiple_comparisons": corrections_filename,
        "optimal_parameters": optimal_filename,
    }


def quick_analysis_report(df, statistical_results):
    """Generate a comprehensive quick analysis report"""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE ANALYSIS REPORT")
    print("=" * 60)

    print(f"📊 STUDY OVERVIEW:")
    print(f"   • Total simulations run: {len(df)}")
    print(f"   • Personality pairings tested: {df['personality_pairing'].nunique()}")
    print(f"   • Parameter combinations tested: {len(df.groupby(['alpha', 'beta']))}")
    print(f"   • Episodes per simulation: 400")

    print(f"\n📈 PERFORMANCE METRICS:")
    print(f"   • Highest total reward: {df['total_reward'].max():.3f}")
    print(f"   • Lowest total reward: {df['total_reward'].min():.3f}")
    print(
        f"   • Average total reward: {df['total_reward'].mean():.3f} ± {df['total_reward'].std():.3f}"
    )
    print(f"   • Most balanced outcome: {df['reward_difference'].min():.3f}")
    print(f"   • Least balanced outcome: {df['reward_difference'].max():.3f}")
    print(
        f"   • Average balance: {df['reward_difference'].mean():.3f} ± {df['reward_difference'].std():.3f}"
    )

    print(f"\n🔬 STATISTICAL SIGNIFICANCE:")
    print(
        f"   • Total statistical tests performed: {len(statistical_results['all_p_values'])}"
    )
    print(
        f"   • Significant after Benjamini-Hochberg correction: {statistical_results['significant_tests_bh']}"
    )
    print(
        f"   • Significant after Bonferroni correction: {statistical_results['significant_tests_bonf']}"
    )

    print(f"\n⚙️ PARAMETER EFFECTS:")
    alpha_effect = statistical_results["alpha_main_effect"]
    beta_effect = statistical_results["beta_main_effect"]
    personality_effect = statistical_results["personality_main_effect"]

    print(
        f"   • Alpha parameter effect: p = {alpha_effect['p_value']:.6f}, η² = {alpha_effect['effect_size']:.4f}"
    )
    print(
        f"   • Beta parameter effect: p = {beta_effect['p_value']:.6f}, η² = {beta_effect['effect_size']:.4f}"
    )
    print(
        f"   • Personality pairing effect: p = {personality_effect['p_value']:.6f}, η² = {personality_effect['effect_size']:.4f}"
    )

    print(f"\n🎯 TOP RECOMMENDATIONS:")

    # Best parameter combination for total reward
    best_total = df.loc[df["total_reward"].idxmax()]
    print(f"   • Best total reward: α={best_total['alpha']}, β={best_total['beta']}")
    print(
        f"     Pairing: {best_total['personality_pairing']}, Score: {best_total['total_reward']:.3f}"
    )

    # Best parameter combination for balance
    best_balance = df.loc[df["reward_difference"].idxmin()]
    print(f"   • Best balance: α={best_balance['alpha']}, β={best_balance['beta']}")
    print(
        f"     Pairing: {best_balance['personality_pairing']}, Difference: {best_balance['reward_difference']:.3f}"
    )

    # Best average performers
    param_avg = df.groupby(["alpha", "beta"])["total_reward"].mean()
    best_avg_params = param_avg.idxmax()
    print(
        f"   • Best average performance: α={best_avg_params[0]}, β={best_avg_params[1]}"
    )
    print(f"     Average score: {param_avg.max():.3f}")


if __name__ == "__main__":
    # Run the enhanced parameter sweep
    results, df, analysis, statistical_results = example_6_interactive_demo()

    # Generate comprehensive report
    quick_analysis_report(df, statistical_results)

    print("\n✅ COMPLETE PARAMETER SWEEP ANALYSIS FINISHED!")
    print("📁 Check the generated CSV files and plots for detailed results.")
    print("🔬 All statistical tests include proper multiple comparison corrections.")
    print("📊 Separate visualizations generated for total reward and balance analysis.")


def create_payoff_visualization():
    """Create standalone payoff landscape visualization."""
    print("=" * 60)
    print("BONUS: Payoff Landscape Visualization")
    print("=" * 60)

    plotter = SimulationPlotter()

    # Create payoff landscapes for different parameter settings
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    parameter_sets = [
        (2.0, 5.0),  # Gentle
        (4.0, 10.0),  # Standard
        (6.0, 15.0),  # Harsh
    ]

    for alpha, beta in parameter_sets:
        print(f"Creating payoff landscape for α={alpha}, β={beta}")
        plotter.plot_payoff_landscape(
            alpha=alpha,
            beta=beta,
            output_dir=f"./payoff_analysis/alpha_{alpha}_beta_{beta}",
        )

    print("Payoff visualizations saved to ./payoff_analysis/")


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        example_num = sys.argv[1]

        if example_num == "all":
            run_all_examples()
        elif example_num == "payoff":
            create_payoff_visualization()
        elif example_num.isdigit():
            example_functions = {
                "1": example_1_basic_interaction,
                "2": example_2_custom_agents,
                "3": example_3_comparison_study,
                "4": example_4_parameter_sweep,
                "5": example_5_longitudinal_analysis,
                "6": example_6_interactive_demo,
            }

            if example_num in example_functions:
                example_functions[example_num]()
            else:
                print(f"Unknown example number: {example_num}")
        else:
            print(f"Unknown command: {example_num}")
    else:
        # Default: run a quick demo
        print("No arguments provided. Running quick demo...")
        print("Use 'python example_usage.py <1-6|all|payoff>' for specific examples")

        control = SACSControlCenter()
        results = control.quick_run("cooperative", "competitive", episodes=200)

        final_eval = results["final_evaluation"]
        print(f"\nQuick Demo Results:")
        print(f"Cooperative: {final_eval['agent1_avg_reward']:.3f}")
        print(f"Competitive: {final_eval['agent2_avg_reward']:.3f}")

        print(f"\nTo run specific examples:")
        print(f"  python example_usage.py 1    # Basic interaction")
        print(f"  python example_usage.py 2    # Custom agents")
        print(f"  python example_usage.py 3    # Comparison study")
        print(f"  python example_usage.py 4    # Parameter sweep")
        print(f"  python example_usage.py 5    # Longitudinal analysis")
        print(f"  python example_usage.py 6    # Interactive demo")
        print(f"  python example_usage.py all  # All examples")
        print(f"  python example_usage.py payoff # Payoff visualizations")
